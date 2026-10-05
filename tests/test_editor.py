import pytest
from core.environment import LocalEnvironment
from core.errors import ToolError
from core.tool import extract_schema
from core.tools.editor import EditorTool


@pytest.fixture
def env(tmp_path):
    return LocalEnvironment(workspace=str(tmp_path))


@pytest.fixture
def editor(env):
    return EditorTool(env=env)


# -----------------------------------------------------------------------------
# LocalEnvironment Tests
# -----------------------------------------------------------------------------


def test_env_path_sandboxing(env):
    with pytest.raises(ToolError) as exc:
        env.read_file("../../etc/passwd")
    assert "escapes workspace boundary" in str(exc.value)


def test_env_read_write_and_exists(env):
    assert not env.exists("test.txt")
    env.write_file("sub/test.txt", "hello parallax")
    assert env.exists("sub/test.txt")
    assert env.is_file("sub/test.txt")
    assert env.is_dir("sub")
    assert env.read_file("sub/test.txt") == "hello parallax"


def test_env_list_dir(env):
    env.write_file("src/main.py", "print('hello')")
    env.write_file("src/.venv/secret.py", "secret")
    env.write_file("docs/readme.md", "docs")

    entries, hidden = env.list_dir(".", max_depth=2)
    assert any("src/main.py" in e for e in entries)
    assert any("docs/readme.md" in e for e in entries)
    assert not any(".venv" in e for e in entries)


def test_env_execute_success(env):
    res = env.execute("echo 'Parallax Env Test'")
    assert res.returncode == 0
    assert "Parallax Env Test" in res.stdout


def test_env_execute_timeout(env):
    with pytest.raises(ToolError) as exc:
        env.execute("sleep 2", timeout=0.1)
    assert "timed out after 0.1 seconds" in str(exc.value)


# -----------------------------------------------------------------------------
# EditorTool Tests
# -----------------------------------------------------------------------------


def test_editor_create_and_view(editor):
    res = editor.str_replace_editor(
        command="create",
        path="app.py",
        file_text="line 1\nline 2\nline 3\nline 4\nline 5\n",
    )
    assert "File created successfully" in res

    # Reject creating already existing file
    with pytest.raises(ToolError) as exc:
        editor.str_replace_editor(command="create", path="app.py", file_text="dup")
    assert "already exists" in str(exc.value)

    # View full file
    view_full = editor.str_replace_editor(command="view", path="app.py")
    assert "     1\tline 1" in view_full
    assert "     5\tline 5" in view_full

    # View range
    view_range = editor.str_replace_editor(
        command="view", path="app.py", view_range=[2, 4]
    )
    assert "line 1" not in view_range
    assert "     2\tline 2" in view_range
    assert "     4\tline 4" in view_range
    assert "line 5" not in view_range


def test_editor_view_directory(editor):
    editor.str_replace_editor(command="create", path="pkg/mod.py", file_text="# mod")
    view_dir = editor.str_replace_editor(command="view", path="pkg")
    assert "pkg/mod.py" in view_dir


def test_editor_str_replace_success_and_preview(editor):
    editor.str_replace_editor(
        command="create",
        path="math.py",
        file_text="def multiply(a, b):\n    return a + b\n",
    )

    replace_res = editor.str_replace_editor(
        command="str_replace",
        path="math.py",
        old_str="return a + b",
        new_str="return a * b",
    )
    assert "Preview of changes:" in replace_res
    assert "     2\t    return a * b" in replace_res

    content = editor.env.read_file("math.py")
    assert "return a * b" in content


def test_editor_str_replace_errors(editor):
    editor.str_replace_editor(
        command="create",
        path="sample.py",
        file_text="val = 10\n",
    )

    # Missing string
    with pytest.raises(ToolError) as exc:
        editor.str_replace_editor(
            command="str_replace", path="sample.py", old_str="non_existent", new_str="val = 20"
        )
    assert "old_str was not found" in str(exc.value)

    # Whitespace mismatch heuristic (old_str has extra whitespace not in file)
    with pytest.raises(ToolError) as exc:
        editor.str_replace_editor(
            command="str_replace", path="sample.py", old_str="   val = 10   ", new_str="val = 20"
        )
    assert "matches when leading/trailing whitespace is stripped" in str(exc.value)

    # Ambiguity check (create a file with duplicate lines)
    editor.str_replace_editor(
        command="create",
        path="dup.py",
        file_text="repeat_me\nrepeat_me\n",
    )
    with pytest.raises(ToolError) as exc:
        editor.str_replace_editor(
            command="str_replace", path="dup.py", old_str="repeat_me", new_str="changed"
        )
    assert "Multiple occurrences (2)" in str(exc.value)
    assert "lines [1, 2]" in str(exc.value)


def test_editor_insert_and_undo(editor):
    editor.str_replace_editor(
        command="create",
        path="ordered.txt",
        file_text="first\nthird\n",
    )

    # Insert
    insert_res = editor.str_replace_editor(
        command="insert",
        path="ordered.txt",
        insert_line=1,
        new_str="second",
    )
    assert "Inserted text after line 1" in insert_res
    assert editor.env.read_file("ordered.txt") == "first\nsecond\nthird\n"

    # Undo
    undo_res = editor.str_replace_editor(command="undo_edit", path="ordered.txt")
    assert "Successfully reverted" in undo_res
    assert editor.env.read_file("ordered.txt") == "first\nthird\n"


def test_editor_schema_extraction(editor):
    schema = extract_schema(editor.str_replace_editor)
    as_dict = schema.to_dict()

    assert as_dict["type"] == "function"
    fn = as_dict["function"]
    assert fn["name"] == "str_replace_editor"

    props = fn["parameters"]["properties"]
    assert props["command"]["type"] == "string"
    assert props["command"]["enum"] == ["view", "create", "str_replace", "insert", "undo_edit"]
    assert props["path"]["type"] == "string"
    assert props["view_range"]["type"] == "array"
    assert props["view_range"]["items"] == {"type": "integer"}

    # command and path are required, the rest are optional
    assert fn["parameters"]["required"] == ["command", "path"]
