"""🔄 带状态的代理

此示例展示了如何创建在交互过程中维持状态的代理。
它演示了一个简单的计数器机制，但这种模式可以扩展到更复杂的状态管理，
如维护对话上下文、用户偏好或跟踪多步骤流程。

要尝试的示例提示：
- "增加计数器3次，并告诉我最终计数"
- "我们当前的计数是多少？再添加2个"
- "让我们增加计数器5次，但每一步都告诉我"
- "给我们的计数加4，并提醒我我们从哪里开始的"
- "增加计数器两次并总结我们的旅程"

运行 `pip install openai agno` 安装依赖。
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import QWQChat

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))


# 定义一个工具，用于增加我们的计数器并返回新值
def increment_counter(agent: Agent) -> str:
    """增加会话计数器并返回新值。"""
    agent.session_state["count"] += 1
    return f"计数现在是 {agent.session_state['count']}"


# 创建一个维护状态的状态管理器代理
agent = Agent(
    model=QWQChat(),
    # 使用从0开始的计数器初始化会话状态
    session_state={"count": 0},
    tools=[increment_counter],
    # 您可以在指令中使用会话状态中的变量
    instructions=dedent("""\
        您是状态管理器，一位热情的状态管理指南！ 🔄
        您的工作是通过简单的计数器示例帮助用户理解状态管理。

        为每次交互遵循以下指南：
        1. 在相关时始终确认当前状态（计数）
        2. 使用increment_counter工具修改状态
        3. 以清晰且引人入胜的方式解释状态变化

        按照以下方式构建您的回复：
        - 当前状态状况
        - 状态转换动作
        - 最终状态和观察结果

        初始状态（计数）是：{count}\
    """),
    show_tool_calls=True,
    add_state_in_messages=True,
    markdown=True,
)

# 使用示例
agent.print_response(
    "让我们增加计数器3次并观察状态变化！",
    stream=True,
)

# 更多要尝试的示例提示：
"""
尝试这些引人入胜的状态管理场景：
1. "更新我们的状态4次并跟踪变化"
2. "修改计数器两次并解释状态转换"
3. "增加3次并显示状态如何持续"
4. "让我们进行5次状态更新并观察"
5. "在我们的计数上增加3并解释状态管理概念"
"""

print(f"最终会话状态：{agent.session_state}")
