from typing import Iterator

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from agno.models.openai import OpenAIChat, QWQChat, \
    DeepSeekV3Chat, QwenMaxChat20250125, QwenV1MaxChat, QwenOmniTurboChat, Qwen2Dot5Omni7bChat

from agno.agent import Agent
from agno.exceptions import RetryAgentRun
from agno.tools import FunctionCall, tool

num_calls = 0


def pre_hook(fc: FunctionCall):
    global num_calls

    print(f"前置钩子: {fc.function.name}")
    print(f"参数: {fc.arguments}")
    num_calls += 1
    if num_calls < 2:
        raise RetryAgentRun(
            "这个不够有趣，请使用不同的参数重试"
        )


@tool(pre_hook=pre_hook)
def print_something(something: str) -> Iterator[str]:
    print(something)
    yield f"我已经打印了 {something}"


agent = Agent(
    model=QwenMaxChat20250125(),
    tools=[print_something],
    markdown=True)
agent.print_response("打印一些有趣的东西", stream=True)
