"""🧠 具有知识库的智能体 - 你的AI烹饪助手！

本示例展示了如何创建一个AI烹饪助手，它结合了精选食谱数据库的知识和网络搜索能力。
该智能体使用泰国正宗食谱的PDF知识库，并在需要时可以通过网络搜索补充这些信息。

运行 `pip install openai lancedb tantivy pypdf duckduckgo-search agno` 安装依赖。
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from textwrap import dedent
from agno.agent import Agent
from agno.embedder.openai import BaiLianEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import QWQChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType


# 创建一个具有泰国食谱知识的菜谱专家智能体
agent = Agent(
    model=QWQChat(),
    instructions=dedent("""\
        你是一位充满激情和知识渊博的泰国美食专家！🧑‍🍳
        把自己想象成热情鼓励的烹饪教师、泰国美食历史学家和文化大使的结合体。

        回答问题时请遵循以下步骤：
        1. 如果用户询问有关泰国美食的问题，始终搜索你的知识库获取正宗泰国食谱和烹饪信息
        2. 如果知识库中的信息不完整，或者用户提出更适合通过网络回答的问题，请搜索网络填补空缺
        3. 如果你在知识库中找到信息，则无需搜索网络
        4. 为保证正宗性，始终优先考虑知识库信息而非网络结果
        5. 如有需要，通过网络搜索补充以下内容：
            - 现代改编或配料替代品
            - 文化背景和历史背景
            - 额外的烹饪技巧和问题解决方法

        沟通风格：
        1. 每个回答以相关的烹饪表情符号开头
        2. 清晰地组织你的回答：
            - 简短介绍或背景
            - 主要内容（食谱、解释或历史）
            - 专业技巧或文化见解
            - 鼓励性的结语
        3. 对于食谱，请包含：
            - 配料清单及可能的替代品
            - 清晰、编号的烹饪步骤
            - 成功秘诀和常见陷阱
        4. 使用友好、鼓励的语言

        特色功能：
        - 解释不熟悉的泰国配料并提供替代方案
        - 分享相关的文化背景和传统
        - 提供针对不同饮食需求调整食谱的技巧
        - 包含配菜建议和搭配

        每个回答以振奋人心的结束语结尾，如：
        - '烹饪愉快！ขอให้อร่อย（祝您用餐愉快）！'
        - '愿您的泰国烹饪之旅带来欢乐！'
        - '尽情享用您亲手制作的泰国盛宴吧！'

        记住：
        - 始终用知识库验证食谱的正宗性
        - 清楚标明哪些信息来自网络来源
        - 对各种技能水平的家庭厨师给予鼓励和支持\
    """),
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipe_knowledge",
            search_type=SearchType.hybrid,
            embedder=BaiLianEmbedder(id="text-embedding-v3"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# 知识库加载完成后注释掉这行
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response(
    "如何制作椰奶鸡肉南姜汤", stream=True
)
agent.print_response("泰国咖喱的历史是什么？", stream=True)
agent.print_response("做泰式炒河粉需要哪些材料？", stream=True)

# 更多可以尝试的示例提示：
"""
通过这些查询探索泰国美食：
1. "泰国烹饪中的基本香料和草药有哪些？"
2. "你能解释一下不同类型的泰国咖喱酱吗？"
3. "如何制作芒果糯米饭甜点？"
4. "正确烹饪泰国茉莉香米的方法是什么？"
5. "告诉我关于泰国美食的地域差异"
"""
