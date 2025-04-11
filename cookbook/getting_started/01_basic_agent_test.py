"""🗽 基本代理示例 - 创建一个独特的新闻记者

此示例展示了如何创建一个具有鲜明个性的基本AI代理。
我们将创建一个结合纽约态度和创意故事讲述的有趣新闻记者。
这展示了个性和风格指令如何塑造代理的响应。

运行 `pip install openai agno` 以安装依赖项。
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import QWQChat

# 创建一个具有有趣个性的新闻记者
agent = Agent(
    model=QWQChat(),
    instructions=dedent("""\
        你是一位热情洋溢、擅长讲故事的新闻记者！🗽
        把自己想象成一个机智的喜剧演员和一个敏锐的记者的结合体。

        你的风格指南：
        - 以引人注目的标题开始，使用表情符号
        - 用热情和纽约态度分享新闻
        - 保持回应简洁但有趣
        - 适当加入当地引用和纽约俚语
        - 以一个吸引人的结束语作为收尾，如"回到演播室！"或"来自大苹果的现场报道！"

        记得在保持纽约能量高涨的同时验证所有事实！\
    """),
    markdown=True,
)

# 示例用法
agent.print_response(
    "告诉我关于时代广场正在发生的一个突发新闻故事。", stream=True
)

# 更多示例提示可以尝试：
"""
尝试这些有趣的场景：
1. "什么是最新席卷布鲁克林的食品潮流？"
2. "告诉我今天在地铁上发生的一个奇特事件"
3. "曼哈顿最新屋顶花园有什么新闻？"
4. "报道一次由动物园逃脱动物引起的不寻常交通堵塞"
5. "报道在中央车站的快闪求婚"
"""
