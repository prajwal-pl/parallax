from core.run import AgentRun
from core.tool import extract_schema
from core.types import AgentResult


class Agent:
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.tool_registry = {}
        self.tool_schemas = []

    def add_tool(self, func):
        schema = extract_schema(func)
        self.tool_registry[func.__name__] = func
        self.tool_schemas.append(schema.to_dict())

    async def run(self, user_input: str) -> AgentResult:
        state = AgentRun(self, user_input)
        return await state.execute()
