"""📰 带上下文的代理

此示例展示如何向代理注入外部依赖项。
上下文在代理运行时被评估，相当于为代理提供依赖注入。

运行 `pip install openai agno` 安装依赖。
"""

import json
from textwrap import dedent

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
import httpx
from agno.agent import Agent
from agno.models.openai import QWQChat


def get_top_hackernews_stories(num_stories: int = 5) -> str:
    """获取并返回HackerNews的热门文章。

    参数：
        num_stories: 要检索的热门文章数量（默认：5）
    返回：
        包含文章详情（标题、URL、评分等）的JSON字符串
    """
    # 获取热门文章
    stories = [
        {
            k: v
            for k, v in httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{id}.json"
            )
            .json()
            .items()
            if k != "kids"  # 排除讨论线程
        }
        for id in httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json"
        ).json()[:num_stories]
    ]
    return json.dumps(stories, indent=4)


# 创建一个可以访问实时HackerNews数据的上下文感知代理
agent = Agent(
    model=QWQChat(),
    # 上下文中的每个函数在代理运行时被评估，
    # 可以将其视为代理的依赖注入
    context={"top_hackernews_stories": get_top_hackernews_stories},
    # add_context将自动将上下文添加到用户消息中
    # add_context=True,
    # 或者，您可以手动将上下文添加到指令中
    instructions=dedent("""\
        您是一位富有洞察力的科技趋势观察者！ 📰

        这是HackerNews上的热门文章：
        {top_hackernews_stories}\
    """),
    markdown=True,
)

# 使用示例
agent.print_response(
    "总结HackerNews上的热门文章并找出任何有趣的趋势。",
    stream=True,
)
