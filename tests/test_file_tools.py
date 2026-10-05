import pytest
from core.errors import ToolError
from core.tools.files import FileTools


def test_file_tools_validate_path_escape(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    with pytest.raises(ToolError) as exc_info:
        tools.validate_path("../outside.txt")
    assert "outside the workspace" in str(exc_info.value)


def test_file_tools_write_and_read(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    
    # Write to nested directory
    res = tools.write_file("nested/dir/hello.txt", "Hello, World!")
    assert "Wrote 13 characters" in res

    # Read back
    content = tools.read_file("nested/dir/hello.txt")
    assert content == "Hello, World!"


def test_file_tools_read_non_existent(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    with pytest.raises(ToolError) as exc_info:
        tools.read_file("does_not_exist.txt")
    assert "File not found" in str(exc_info.value)


def test_file_tools_edit_file_exact_match(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    tools.write_file("code.py", "def foo():\n    return 1\n")

    res = tools.edit_file("code.py", old="return 1", new="return 42")
    assert "Successfully edited" in res

    content = tools.read_file("code.py")
    assert "return 42" in content


def test_file_tools_edit_file_not_found(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    tools.write_file("code.py", "def foo():\n    return 1\n")

    with pytest.raises(ToolError) as exc_info:
        tools.edit_file("code.py", old="return 999", new="return 42")
    assert "Text to replace was not found" in str(exc_info.value)


def test_file_tools_edit_file_ambiguous_multiple_matches(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    tools.write_file("repeat.txt", "abc\nabc\n")

    with pytest.raises(ToolError) as exc_info:
        tools.edit_file("repeat.txt", old="abc", new="xyz")
    assert "Found 2 occurrences" in str(exc_info.value)


def test_file_tools_list_files(tmp_path):
    tools = FileTools(workspace=str(tmp_path))
    tools.write_file("app/main.py", "print(1)")
    tools.write_file("app/.venv/secret.py", "secret")
    tools.write_file(".git/config", "git")

    listing = tools.list_files()
    assert "app/main.py" in listing
    # Ignored directories should not be listed
    assert "secret.py" not in listing
    assert ".git" not in listing
