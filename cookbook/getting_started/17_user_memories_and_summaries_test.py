"""🧠 长期用户记忆和会话摘要

此示例展示了如何创建具有持久记忆的代理，该记忆存储：
1. 个性化用户记忆 - 关于特定用户学到的事实和偏好
2. 会话摘要 - 来自对话的关键点和上下文
3. 聊天历史 - 存储在SQLite中以实现持久性

关键功能：
- 在SQLite数据库中存储用户特定记忆
- 维护会话摘要以提供上下文
- 跨会话使用记忆继续对话
- 在回复中引用先前的上下文和用户信息

示例：
用户："我叫John，住在纽约"
代理：*创建关于John位置的记忆*

用户："你对我有什么记忆？"
代理：*回忆起关于John的先前记忆*

运行：`pip install openai sqlalchemy agno` 安装依赖
"""

import json
from textwrap import dedent
from typing import Optional

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

import typer
from agno.agent import Agent, AgentMemory
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.models.openai import OpenAIChat, QWQChat, \
    DeepSeekV3Chat, QwenMaxChat20250125, QwenV1MaxChat, QwenOmniTurboChat, Qwen2Dot5Omni7bChat
from agno.storage.sqlite import SqliteStorage
from agno.memory.manager import MemoryManager
from agno.memory.classifier import MemoryClassifier
from agno.memory.summarizer import MemorySummarizer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt


os.environ["OPENAI_API_KEY"] = "sk-df66e1c0892143e7a610c5c20db08d0e"

def create_agent(user: str = "user"):
    session_id: Optional[str] = None

    # 询问用户是否要开始新会话或继续现有会话
    new = typer.confirm("您想开始新会话吗？")

    # 为代理会话和记忆初始化存储
    agent_storage = SqliteStorage(table_name="agent_memories", db_file="tmp/agents.db")

    if not new:
        existing_sessions = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        model=QwenMaxChat20250125(),
        user_id=user,
        session_id=session_id,
        # 使用SQLite存储配置记忆系统
        memory=AgentMemory(
            manager=MemoryManager(model=QwenMaxChat20250125()),
            classifier=MemoryClassifier(model=QwenMaxChat20250125()),
            summarizer=MemorySummarizer(model=QwenMaxChat20250125()),
            db=SqliteMemoryDb(
                table_name="agent_memory",
                db_file="tmp/agent_memory.db",
            ),
            create_user_memories=True,
            update_user_memories_after_run=True,
            create_session_summary=True,
            update_session_summary_after_run=True,
        ),
        storage=agent_storage,
        add_history_to_messages=True,
        num_history_responses=3,
        # 增强系统提示以获得更好的个性和记忆使用
        description=dedent("""\
        您是一个具有出色记忆力的乐于助人且友好的AI助手。
        - 记住关于用户的重要细节并自然地引用它们
        - 保持温暖、积极的语调，同时保持精确和乐于助人
        - 在适当时，引用先前的对话和记忆
        - 对于您记得或不记得的内容始终保持诚实"""),
        markdown=True,
    )

    if session_id is None:
        session_id = agent.session_id
        if session_id is not None:
            print(f"已开始会话: {session_id}\n")
        else:
            print("已开始会话\n")
    else:
        print(f"继续会话: {session_id}\n")

    return agent


def print_agent_memory(agent):
    """打印代理记忆系统的当前状态"""
    console = Console()

    # 打印聊天历史
    console.print(
        Panel(
            JSON(
                json.dumps([m.to_dict() for m in agent.memory.messages]),
                indent=4,
            ),
            title=f"会话ID的聊天历史: {agent.session_id}",
            expand=True,
        )
    )

    # 打印用户记忆
    console.print(
        Panel(
            JSON(
                json.dumps(
                    [
                        m.model_dump(include={"memory", "input"})
                        for m in agent.memory.memories
                    ]
                ),
                indent=4,
            ),
            title=f"用户ID的记忆: {agent.user_id}",
            expand=True,
        )
    )

    # 打印会话摘要
    console.print(
        Panel(
            JSON(json.dumps(agent.memory.summary.model_dump(), indent=4)),
            title=f"会话ID的摘要: {agent.session_id}",
            expand=True,
        )
    )


def main(user: str = "user"):
    """带记忆显示的交互式聊天循环"""
    agent = create_agent(user)

    print("尝试这些示例输入：")
    print("- '我的名字是[名字]，我住在[城市]'")
    print("- '我喜欢[爱好/兴趣]'")
    print("- '你对我记得什么？'")
    print("- '我们迄今为止讨论了什么？'\n")

    exit_on = ["exit", "quit", "bye"]
    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in exit_on:
            break

        agent.print_response(message=message, stream=True, markdown=True)
        print_agent_memory(agent)


if __name__ == "__main__":
    typer.run(main)
