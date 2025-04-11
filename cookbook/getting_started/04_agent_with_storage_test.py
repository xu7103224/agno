"""🧠 带存储的代理 - 您的AI泰国烹饪助手！

此示例展示了如何创建一个AI烹饪助手，它结合了精心策划的食谱数据库和网络搜索功能。
该代理使用正宗泰国食谱的PDF知识库，并可在需要时用网络搜索补充这些信息。

要尝试的示例提示：
- "如何制作正宗的泰式炒河粉？"
- "红咖喱和绿咖喱有什么区别？"
- "你能解释一下南姜是什么以及可能的替代品吗？"
- "告诉我关于冬荫功汤的历史"
- "泰国食品储藏室的必备食材有哪些？"
- "如何制作泰国罗勒鸡肉（Pad Kra Pao）？"

运行 `pip install openai lancedb tantivy pypdf duckduckgo-search sqlalchemy agno` 安装依赖。
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from textwrap import dedent
from typing import List, Optional

import typer
from agno.agent import Agent
from agno.embedder.openai import BaiLianEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import QWQChat
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType
from rich import print

agent_knowledge = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="recipe_knowledge",
        search_type=SearchType.hybrid,
        embedder=BaiLianEmbedder(),
    ),
)
# 加载知识库后注释掉
# if agent_knowledge is not None:
#     agent_knowledge.load()

agent_storage = SqliteStorage(table_name="recipe_agent", db_file="tmp/agents.db")


def recipe_agent(user: str = "user"):
    session_id: Optional[str] = None

    # 询问用户是否要开始新会话或继续现有会话
    new = typer.confirm("您想开始新会话吗？")

    if not new:
        existing_sessions: List[str] = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        user_id=user,
        session_id=session_id,
        model=QWQChat(),
        instructions=dedent("""\
            您是一位热情且知识渊博的泰国料理专家！ 🧑‍🍳
            把自己想象成一个温暖、鼓励人的烹饪指导员、
            泰国美食历史学家和文化大使的结合体。

            回答问题时请遵循以下步骤：
            1. 首先，在知识库中搜索正宗泰国食谱和烹饪信息
            2. 如果知识库中的信息不完整或者用户提出的问题更适合网络搜索，请搜索网络填补空白
            3. 如果您在知识库中找到了信息，则无需搜索网络
            4. 为了保证真实性，始终优先考虑知识库信息而非网络结果
            5. 如有需要，通过网络搜索补充：
               - 现代改编或原料替代品
               - 文化背景和历史背景
               - 额外的烹饪技巧和疑难解答

            沟通风格：
            1. 在每个回答开始使用相关烹饪表情符号
            2. 清晰地构建您的回答：
               - 简短介绍或背景
               - 主要内容（食谱、解释或历史）
               - 专业提示或文化见解
               - 鼓励性结论
            3. 对于食谱，包括：
               - 配料清单和可能的替代品
               - 清晰、编号的烹饪步骤
               - 成功技巧和常见陷阱
            4. 使用友好、鼓励的语言

            特殊功能：
            - 解释不熟悉的泰国原料并建议替代品
            - 分享相关文化背景和传统
            - 提供适应不同饮食需求的食谱技巧
            - 包括配菜建议和配菜

            以鼓舞人心的告别语结束每个回答，如：
            - '烹饪愉快！ขอให้อร่อย（祝您好胃口）！'
            - '愿您的泰国烹饪冒险带来快乐！'
            - '享用您自制的泰国盛宴！'

            请记住：
            - 始终用知识库验证食谱的真实性
            - 清楚说明信息何时来自网络来源
            - 对所有技能水平的家庭厨师给予鼓励和支持\
        """),
        storage=agent_storage,
        knowledge=agent_knowledge,
        tools=[DuckDuckGoTools()],
        # 在响应中显示工具调用
        show_tool_calls=True,
        # 为代理提供聊天历史
        # 我们可以：
        # 1. 为代理提供读取聊天历史的工具
        # 2. 自动将聊天历史添加到发送给模型的消息中
        #
        # 1. 为代理提供读取聊天历史的工具
        read_chat_history=True,
        # 2. 自动将聊天历史添加到发送给模型的消息中
        # add_history_to_messages=True,
        # 添加到消息中的历史响应数量。
        # num_history_responses=3,
        markdown=True,
    )

    print("您将要与代理聊天！")
    if session_id is None:
        session_id = agent.session_id
        if session_id is not None:
            print(f"已开始会话: {session_id}\n")
        else:
            print("已开始会话\n")
    else:
        print(f"继续会话: {session_id}\n")

    # 将代理作为命令行应用程序运行
    agent.cli_app(markdown=True, stream=True)


if __name__ == "__main__":
    typer.run(recipe_agent)
