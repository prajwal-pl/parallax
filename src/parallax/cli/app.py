from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


from parallax import (
    Agent,
    BashTool,
    ContextConfig,
    EditorTool,
    EventType,
    LocalEnvironment,
    SearchTool,
)
from parallax.cli.console import get_console
from parallax.cli.doctor import run_doctor
from parallax.cli.renderer import CliRenderer
from parallax.cli.repl import InteractiveRepl
from parallax.cli.theme import ICON_ERROR, ICON_SUCCESS, ICON_WARNING, PANEL_BOX

from parallax.cli.config import (
    get_provider_status,
    has_configured_keys,
    load_global_config,
    prompt_onboarding,
    set_provider_key,
)

# Ensure environment variables are loaded (.env and ~/.parallax/config.json)
load_dotenv()
load_global_config()


def get_default_model() -> str:
    """Determine the default model based on environment variables and available API keys."""
    if os.environ.get("PARALLAX_MODEL"):
        return os.environ["PARALLAX_MODEL"]
    if os.environ.get("PARALLAX_PROVIDER", "").lower() == "groq":
        return "llama-3.3-70b-versatile"
    # Auto-fallback to Groq if user only configured GROQ_API_KEY
    if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        return "llama-3.3-70b-versatile"
    return "liquid/lfm-2.5-2.6b:free"


DEFAULT_MODEL = "liquid/lfm-2.5-2.6b:free"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert autonomous software engineer. "
    "You have access to:\n"
    "- str_replace_editor: view, create, and modify files\n"
    "- bash: run tests, linters, and shell commands\n"
    "- search_code: search text and regex patterns across the codebase\n"
    "Investigate thoroughly, inspect errors, modify files carefully, and verify changes."
)

app = typer.Typer(
    name="parallax",
    help="Modular, model-agnostic AI coding agent runtime.",
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def build_default_agent(
    model: str | None = None,
    system_prompt: str | None = None,
    workspace: str | Path | None = None,
) -> Agent:
    """Create a fully equipped Parallax Agent with core tools registered."""
    ws = str(workspace or Path.cwd())
    env = LocalEnvironment(workspace=ws)
    editor = EditorTool(env=env)
    bash_tool = BashTool(env=env)
    search_tool = SearchTool(env=env)

    chosen_model = model or get_default_model()
    agent = Agent(
        model=chosen_model,
        context=ContextConfig(workspace=ws),
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )

    agent.add_tool(editor.str_replace_editor)
    agent.add_tool(bash_tool.bash)
    agent.add_tool(search_tool.search_code)

    return agent


async def _run_single_task(
    agent: Agent,
    task: str,
    verbose: bool = False,
    quiet: bool = False,
) -> int:
    """Execute a single task and return exit code (0 for success, 1 for failure)."""
    console = get_console()
    renderer = CliRenderer(console=console, verbose=verbose, quiet=quiet)
    exit_code = 0

    try:
        async for event in agent.run_stream(task):
            renderer.handle_event(event)
            if event.type == EventType.RUN_FAILED:
                exit_code = 1
    except KeyboardInterrupt, asyncio.CancelledError:
        renderer.stop_spinner()
        console.print("\n  [status.warning]! run cancelled[/status.warning]\n")
        return 130
    except Exception as e:
        renderer.stop_spinner()
        console.print(f"\n  [status.error]✕ execution error: {e}[/status.error]\n")
        return 1

    return exit_code


@app.callback(invoke_without_command=True)
def default_entry(
    ctx: typer.Context,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model identifier to use."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show expanded tool outputs."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only output the final answer."),
    ] = False,
):
    """
    Parallax AI coding agent CLI.

    Execute a single task directly or launch an interactive session.
    """
    if ctx.invoked_subcommand is not None:
        return

    if not has_configured_keys():
        prompt_onboarding()

    agent = build_default_agent(model=model)
    task_args = " ".join(ctx.args).strip()

    if task_args:
        # Single-shot execution from positional arguments
        code = asyncio.run(
            _run_single_task(agent, task_args, verbose=verbose, quiet=quiet)
        )
        if code != 0:
            raise typer.Exit(code=code)
    else:
        # Launch interactive REPL
        repl = InteractiveRepl(agent=agent, verbose=verbose)
        asyncio.run(repl.run())


@app.command("chat")
def chat_cmd(
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model identifier to use."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show expanded tool outputs."),
    ] = False,
):
    """Start an interactive multi-turn chat session with Parallax."""
    if not has_configured_keys():
        prompt_onboarding()
    agent = build_default_agent(model=model)
    repl = InteractiveRepl(agent=agent, verbose=verbose)
    asyncio.run(repl.run())


@app.command("run")
def run_cmd(
    task: Annotated[str, typer.Argument(help="Task instruction to execute.")],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model identifier to use."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show expanded tool outputs."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only output the final answer."),
    ] = False,
):
    """Execute a single task and return on completion."""
    if not has_configured_keys():
        prompt_onboarding()
    agent = build_default_agent(model=model)

    code = asyncio.run(_run_single_task(agent, task, verbose=verbose, quiet=quiet))
    if code != 0:
        raise typer.Exit(code=code)


@app.command("doctor")
def doctor_cmd():
    """Run environment, configuration, and connectivity diagnostics."""
    healthy = run_doctor()
    if not healthy:
        raise typer.Exit(code=1)


@app.command("tools")
def tools_cmd():
    """List all available tools registered in the runtime."""
    agent = build_default_agent()
    console = get_console()

    pills = Text()
    pills.append(" Tools ", style="bold white on #4f46e5")
    pills.append(f"  {len(agent.tool_schemas)} active runtime tools  ", style="bold #93c5fd on #1e293b")
    console.print()
    console.print(pills)
    console.print()

    for schema in agent.tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "").splitlines()[0] if func.get("description") else ""
        params = func.get("parameters", {}).get("properties", {})

        content = Text()
        content.append(f"{desc}\n\n", style="muted")
        content.append("Parameters: ", style="bold #94a3b8")
        if params:
            for param_name in params.keys():
                content.append(f" {param_name} ", style="bold #93c5fd on #1e293b")
                content.append(" ")
        else:
            content.append("none", style="dim")
        content.append("\n")

        panel = Panel(
            content,
            title=f"[bold #818cf8]{name}[/bold #818cf8]",
            title_align="left",
            box=PANEL_BOX,
            border_style="#334155",
            padding=(0, 2),
        )
        console.print(panel)
    console.print()


@app.command("config")
def config_cmd(
    action: Annotated[
        Optional[str],
        typer.Argument(help="Action: 'show'/'status', 'set', 'setup', or provider name ('groq', 'openrouter')."),
    ] = None,
    target: Annotated[
        Optional[str],
        typer.Argument(help="Provider name when using 'set', or API key when provider is first argument."),
    ] = None,
    key: Annotated[
        Optional[str],
        typer.Argument(help="API key string when using 'set <provider> <key>'."),
    ] = None,
):
    """View or configure LLM provider API keys and runtime settings."""
    console = get_console()

    # Determine command invocation pattern
    provider = None
    api_key = None

    if action in ("setup", "wizard", "init"):
        prompt_onboarding(console=console)
        return

    if action == "set":
        provider = target
        api_key = key
    elif action in ("groq", "openrouter", "or", "g"):
        provider = action
        api_key = target
    elif action in (None, "show", "status", "list"):
        # Display current status table
        pass
    else:
        provider = action
        api_key = target

    if provider and api_key:
        if set_provider_key(provider, api_key):
            console.print(
                f"\n  [status.success]{ICON_SUCCESS}[/status.success] "
                f"[muted]{provider.capitalize()} API Key saved to[/muted] [tool.name]~/.parallax/config.json[/tool.name].\n"
            )
            return
        else:
            console.print(
                f"\n  [status.error]{ICON_ERROR}[/status.error] "
                f"[muted]Unknown provider '{provider}'. Choose 'groq' or 'openrouter'.[/muted]\n"
            )
            raise typer.Exit(code=1)

    # Show configuration table
    statuses = get_provider_status()
    table = Table(
        box=PANEL_BOX,
        border_style="border",
        show_header=True,
        header_style="bold #818cf8",
        padding=(0, 2),
    )
    table.add_column("Provider", style="tool.name")
    table.add_column("Status", style="status.info")
    table.add_column("Key", style="white")
    table.add_column("Default Model", style="dim")
    table.add_column("Description", style="muted")

    for k, p in statuses.items():
        status_txt = (
            f"[status.success]{ICON_SUCCESS} configured[/status.success]"
            if p["configured"]
            else f"[status.warning]{ICON_WARNING} unset[/status.warning]"
        )
        table.add_row(
            p["name"],
            status_txt,
            p["masked"],
            p["default_model"],
            p["description"],
        )

    console.print()
    console.print(
        Panel(
            table,
            title="[brand]parallax config[/brand] [muted]· inference providers[/muted]",
            title_align="left",
            box=PANEL_BOX,
            border_style="#334155",
            padding=(0, 1),
        )
    )
    console.print("[dim #64748b]To set a key: parallax config set <groq|openrouter> <api-key>[/dim #64748b]")
    console.print("[dim #64748b]To run the setup wizard: parallax config setup[/dim #64748b]\n")


@app.command("key")
def key_cmd(
    provider: Annotated[
        Optional[str],
        typer.Argument(help="Provider name ('groq' or 'openrouter') or 'set'."),
    ] = None,
    target: Annotated[
        Optional[str],
        typer.Argument(help="API key or provider name."),
    ] = None,
    key: Annotated[
        Optional[str],
        typer.Argument(help="API key string if using 'parallax key set <prov> <key>'."),
    ] = None,
):
    """Alias for 'parallax config' to manage API keys."""
    config_cmd(action=provider, target=target, key=key)


@app.command("version")
def version_cmd():
    """Show Parallax version and runtime information."""
    console = get_console()
    console.print("[brand]parallax[/brand] [muted]v0.1.3[/muted]")


def main():
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
