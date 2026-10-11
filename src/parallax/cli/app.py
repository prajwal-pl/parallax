from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich.table import Table

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
from parallax.cli.theme import PANEL_BOX

# Ensure .env is loaded
load_dotenv()

DEFAULT_MODEL = os.environ.get("PARALLAX_MODEL", "liquid/lfm-2.5-2.6b:free")
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

    chosen_model = model or DEFAULT_MODEL
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

    table = Table(
        title="[brand]parallax tools[/brand] [muted]· active runtime tools[/muted]",
        title_justify="left",
        box=PANEL_BOX,
        border_style="border",
        show_header=True,
        header_style="bold #818cf8",
        padding=(0, 1),
    )
    table.add_column("name", style="tool.name", no_wrap=True)
    table.add_column("description", style="muted")
    table.add_column("parameters", style="meta")

    for schema in agent.tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "none"
        table.add_row(name, desc.splitlines()[0] if desc else "", param_names)

    console.print()
    console.print(table)
    console.print()


@app.command("version")
def version_cmd():
    """Show Parallax version and runtime information."""
    console = get_console()
    console.print("[brand]parallax[/brand] [muted]v0.1.0[/muted]")


def main():
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
