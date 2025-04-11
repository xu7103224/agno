"""🗣️ 持久聊天历史，即会话记忆

此示例展示了如何创建一个具有存储在SQLite数据库中的持久记忆的代理。
当恢复对话时，我们在代理上设置session_id，这样以前的聊天历史就被保留了。

关键功能：
- 在SQLite数据库中存储对话历史
- 跨多个会话继续对话
- 在回复中引用前期上下文

运行 `pip install openai sqlalchemy agno` 安装依赖。
"""

import json
from typing import Optional

import typer
from agno.agent import Agent
from agno.models.openai import QWQChat
from agno.storage.sqlite import SqliteStorage

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from rich import print
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def create_agent(user: str = "user"):
    session_id: Optional[str] = None

    # 询问用户是否要开始新会话或继续现有会话
    new = typer.confirm("您想开始新会话吗？")

    # 如果用户不想要新会话，则获取现有会话
    agent_storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agents.db")

    if not new:
        existing_sessions = agent_storage.get_all_session_ids(user)
        if len(existing_sessions) > 0:
            session_id = existing_sessions[0]

    agent = Agent(
        user_id=user,
        # 在代理上设置session_id以恢复对话
        session_id=session_id,
        model=QWQChat(),
        storage=agent_storage,
        # 将聊天历史添加到消息中
        add_history_to_messages=True,
        num_history_responses=3,
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


def print_messages(agent):
    """在格式化面板中打印当前聊天历史"""
    console.print(
        Panel(
            JSON(
                json.dumps(
                    [
                        m.model_dump(include={"role", "content"})
                        for m in agent.memory.messages
                    ]
                ),
                indent=4,
            ),
            title=f"会话ID的聊天历史: {agent.session_id}",
            expand=True,
        )
    )


def main(user: str = "user"):
    agent = create_agent(user)

    print("与OpenAI代理聊天！")
    exit_on = ["exit", "quit", "bye"]
    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in exit_on:
            break

        agent.print_response(message=message, stream=True, markdown=True)
        print_messages(agent)


if __name__ == "__main__":
    typer.run(main)
