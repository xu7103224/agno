"""🤝 人机协作：为工具调用添加用户确认功能

此示例展示了如何在您的Agno工具中实现人机协作功能。
它展示了如何：
- 为工具添加前置钩子以获取用户确认
- 在工具执行过程中处理用户输入
- 根据用户选择优雅地取消操作

一些实际应用：
- 在执行敏感操作前进行确认
- 在API调用前进行审查
- 验证数据转换
- 在关键系统中批准自动化操作

运行 `pip install openai httpx rich agno` 安装依赖。
"""

import json
from textwrap import dedent
from typing import Iterator

import httpx
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from agno.models.openai import QWQChat
from agno.agent import Agent
from agno.exceptions import StopAgentRun
from agno.tools import FunctionCall, tool
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Prompt

# 这是由print_response方法使用的控制台实例
# 我们可以用它来停止和重新启动实时显示并请求用户确认
console = Console()


def pre_hook(fc: FunctionCall):
    # 从控制台获取实时显示实例
    live = console._live

    # 临时停止实时显示，以便我们可以请求用户确认
    live.stop()  # type: ignore

    # 请求确认
    console.print(f"\n即将运行 [bold blue]{fc.function.name}[/]")
    message = (
        Prompt.ask("您想继续吗？", choices=["y", "n"], default="y")
        .strip()
        .lower()
    )

    # 重新启动实时显示
    live.start()  # type: ignore

    # 如果用户不想继续，抛出StopExecution异常
    if message != "y":
        raise StopAgentRun(
            "用户取消了工具调用",
            agent_message="由于未获得许可，停止执行。",
        )


@tool(pre_hook=pre_hook)
def get_top_hackernews_stories(num_stories: int) -> Iterator[str]:
    """获取用户确认后的Hacker News热门文章。

    参数：
        num_stories (int): 要获取的文章数量

    返回：
        str: 包含文章详情的JSON字符串
    """
    # 获取热门文章ID
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # 生成文章详情
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        yield json.dumps(story)



# 使用技术达人个性和清晰指令初始化代理
agent = Agent(
    model=QWQChat(),
    description="一个获取并总结Hacker News文章的技术新闻助手",
    instructions=dedent("""\
        你是一个热情洋溢的技术报道员

        你的责任：
        - 以引人入胜且信息丰富的方式呈现Hacker News文章
        - 清晰地总结你收集到的信息

        风格指南：
        - 使用emoji使你的回复更加吸引人
        - 保持简洁但内容丰富的总结
        - 以友好的技术主题结束语作为结尾\
    """),
    tools=[get_top_hackernews_stories],
    show_tool_calls=True,
    markdown=True,
)

# 尝试的示例问题：
# - "现在Hacker News上的前3篇文章是什么？"
# - "向我展示Hacker News最近的一篇文章"
# - "获取前5篇文章（您可以尝试接受和拒绝确认）"
agent.print_response(
    "Hacker News上的前2篇文章是什么？", stream=True, console=console
)

# 查看所有消息
pprint(agent.run_response.messages)
