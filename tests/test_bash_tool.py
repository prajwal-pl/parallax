import pytest
from core.environment import LocalEnvironment
from core.errors import ToolError
from core.tool import extract_schema
from core.tools.bash import BashTool


@pytest.fixture
def bash_tool(tmp_path):
    env = LocalEnvironment(workspace=str(tmp_path))
    return BashTool(env=env)


def test_bash_execution_success(bash_tool):
    res = bash_tool.bash("echo 'Hello Parallax'")
    assert "exit code 0" in res
    assert "Hello Parallax" in res


def test_bash_execution_failure(bash_tool):
    res = bash_tool.bash("false")
    assert "failed (exit code 1)" in res


def test_bash_captures_stderr(bash_tool):
    res = bash_tool.bash("echo 'error log' >&2")
    assert "error log" in res


def test_bash_no_output(bash_tool):
    res = bash_tool.bash("true")
    assert "succeeded (exit code 0) with no output" in res


def test_bash_timeout(bash_tool):
    with pytest.raises(ToolError) as exc:
        bash_tool.bash("sleep 2", timeout=0.1)
    assert "timed out after 0.1 seconds" in str(exc.value)


def test_bash_schema_extraction(bash_tool):
    schema = extract_schema(bash_tool.bash)
    as_dict = schema.to_dict()

    assert as_dict["type"] == "function"
    fn = as_dict["function"]
    assert fn["name"] == "bash"

    props = fn["parameters"]["properties"]
    assert props["command"]["type"] == "string"
    assert props["timeout"]["type"] == "number"
    assert props["timeout"]["default"] == 30.0

    assert fn["parameters"]["required"] == ["command"]
