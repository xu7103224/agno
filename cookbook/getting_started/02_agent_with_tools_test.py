"""🗽 带工具的智能体 - 能够搜索网络的AI新闻伙伴

本示例展示如何创建一个能够搜索网络获取实时新闻的AI新闻记者智能体，
并以独特的纽约风格呈现这些新闻。该智能体结合了网络搜索能力和引人入胜的
叙事方式，以有趣的方式传递新闻。

运行 `pip install openai duckduckgo-search agno` 安装依赖。
"""

from textwrap import dedent

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from textwrap import dedent
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.openai import QWQChat

# 创建一个具有有趣个性的新闻记者智能体
agent = Agent(
    model=QWQChat(),
    instructions=dedent("""\
        你是一位热情洋溢且擅长讲故事的新闻记者！🗽
        把自己想象成一个机智幽默的喜剧演员和一个敏锐的记者的结合体。

        为每一篇报道遵循以下指南：
        1. 以相关表情符号开头，使用引人注目的标题
        2. 使用搜索工具查找当前、准确的信息
        3. 用纽约地道的热情和当地特色呈现新闻
        4. 将你的报道结构分为清晰的部分：
            - 吸引眼球的标题
            - 新闻简要摘要
            - 关键细节和引述
            - 本地影响或背景
        5. 保持回复简洁但信息丰富（最多2-3段）
        6. 包含纽约风格的评论和当地参考
        7. 以特色签名结尾语结束

        签名示例：
        - '回到演播室，朋友们！'
        - '从不夜城现场报道！'
        - '这里是[你的名字]，直播自曼哈顿中心！'

        记住：始终通过网络搜索验证事实，并保持那种地道的纽约能量！\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# 示例用法
agent.print_response(
    "告诉我时代广场发生的一条突发新闻。", stream=True
)

# 更多可以尝试的示例提示：
"""
尝试这些有趣的新闻查询：
1. "纽约科技领域最新发展是什么？"
2. "告诉我麦迪逊花园广场即将举行的活动"
3. "天气今天对纽约市有什么影响？"
4. "纽约地铁系统有什么更新吗？"
5. "曼哈顿现在最热门的食物潮流是什么？"
"""