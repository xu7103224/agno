"""🛠️ 编写您自己的工具 - 使用Hacker News API的示例

此示例展示了如何用Agno创建和使用您自己的自定义工具。
您可以用任何API或服务替换Hacker News功能！

自己工具的一些想法：
- 天气数据获取器
- 股价分析器
- 个人日历集成
- 自定义数据库查询
- 本地文件操作

运行 `pip install openai httpx agno` 安装依赖。
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


def get_top_hackernews_stories(num_stories: int = 10) -> str:
    """使用此函数从Hacker News获取热门故事。

    参数：
        num_stories (int): 要返回的故事数量。默认为10。

    返回：
        str: 热门故事的JSON字符串。
    """

    # 获取热门故事ID
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # 获取故事详情
    stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        stories.append(story)
    return json.dumps(stories)


# 创建一个具有硅谷个性的科技新闻记者代理
agent = Agent(
    model=QWQChat(),
    instructions=dedent("""\
        您是一位精通技术的Hacker News记者，对所有与技术相关的事物充满热情！ 🤖
        把自己想象成硅谷内部人士和科技记者的结合体。

        您的风格指南：
        - 以引人注目的科技标题和emoji开始
        - 以热情和前沿技术态度呈现Hacker News故事
        - 保持回复简洁但内容丰富
        - 在适当时使用科技行业参考和创业术语
        - 以吸引人的科技主题结束语结尾，如"回到终端！"或"推送到生产环境！"

        记得在保持技术热情的同时彻底分析HN故事！\
    """),
    tools=[get_top_hackernews_stories],
    show_tool_calls=True,
    markdown=True,
)

# 要尝试的示例问题：
# - "HN上现在有哪些趋势性技术讨论？"
# - "总结Hacker News上的前5个故事"
# - "今天点赞最多的故事是什么？"
agent.print_response("总结hackernews上的前5个故事？", stream=True)
