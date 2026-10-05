class ParallaxError(Exception):
    pass


class ToolError(ParallaxError):
    pass


class MaxIterationsError(ParallaxError):
    pass


class ToolExecutionError(ParallaxError):
    pass


class ToolNotFoundError(ParallaxError):
    pass


class LLMExecutionError(ParallaxError):
    pass
