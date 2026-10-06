import pytest
from core.environment import LocalEnvironment
from core.errors import ToolError
from core.tool import extract_schema
from core.tools.search import SearchTool


@pytest.fixture
def search_tool(tmp_path):
    env = LocalEnvironment(workspace=str(tmp_path))
    env.write_file("src/math.py", "def add(a, b):\n    return a + b\n")
    env.write_file("src/utils.py", "def helper():\n    return 'ADD_MODE'\n")
    env.write_file("docs/readme.txt", "This is documentation.\n")
    return SearchTool(env=env)


def test_search_code_case_insensitive(search_tool):
    res = search_tool.search_code(query="add")
    assert "src/math.py:1: def add(a, b):" in res
    assert "src/utils.py:2: return 'ADD_MODE'" in res
    assert "Found 2 matches" in res


def test_search_code_case_sensitive(search_tool):
    res = search_tool.search_code(query="ADD_MODE", case_sensitive=True)
    assert "src/utils.py:2: return 'ADD_MODE'" in res
    assert "src/math.py" not in res
    assert "Found 1 match" in res


def test_search_code_specific_directory(search_tool):
    res = search_tool.search_code(query="add", path="docs")
    assert "No matches found" in res


def test_search_code_single_file(search_tool):
    res = search_tool.search_code(query="documentation", path="docs/readme.txt")
    assert "docs/readme.txt:1: This is documentation." in res


def test_search_code_non_existent_path(search_tool):
    with pytest.raises(ToolError) as exc:
        search_tool.search_code(query="test", path="non_existent")
    assert "Path does not exist" in str(exc.value)


def test_search_code_regex(search_tool):
    res = search_tool.search_code(query=r"def \w+\(")
    assert "src/math.py:1: def add(a, b):" in res
    assert "src/utils.py:1: def helper():" in res


def test_search_code_schema_extraction(search_tool):
    schema = extract_schema(search_tool.search_code)
    as_dict = schema.to_dict()

    assert as_dict["type"] == "function"
    fn = as_dict["function"]
    assert fn["name"] == "search_code"

    props = fn["parameters"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["path"]["type"] == "string"
    assert props["case_sensitive"]["type"] == "boolean"
    assert props["max_results"]["type"] == "integer"

    assert fn["parameters"]["required"] == ["query"]
