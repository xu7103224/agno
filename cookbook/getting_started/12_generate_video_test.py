"""🎥 使用ModelsLabs生成视频 - 使用Agno创建AI视频

此示例展示了如何创建一个使用ModelsLabs生成视频的AI代理。
您可以使用此代理创建各种类型的短视频，从动画场景到创意视觉故事。

要尝试的示例提示：
- "创建一个在日落时海浪拍打沙滩的宁静视频"
- "生成一个蝴蝶在魔法森林中飞舞的神奇视频"
- "创建一个花朵在花园中绽放的延时摄影"
- "生成一个北极光在夜空中舞动的视频"

运行 `pip install openai agno` 安装依赖。
记得在环境变量 `MODELS_LAB_API_KEY` 中设置您的ModelsLabs API密钥。
"""

from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import QWQChat
from agno.tools.models_labs import ModelsLabTools

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

# 创建创意AI视频导演代理
video_agent = Agent(
    model=QWQChat(),
    tools=[ModelsLabTools()],
    description=dedent("""\
        您是一位经验丰富的AI视频导演，擅长各种视频风格，
        从自然场景到艺术动画。您对运动、时间和通过视频内容进行视觉讲故事有深刻理解。\
    """),
    instructions=dedent("""\
        作为AI视频导演，请遵循以下指南：
        1. 仔细分析用户的请求，了解所需的风格和情绪
        2. 在生成之前，用关于动作、时间和氛围的细节丰富提示
        3. 使用具有详细、精心制作的提示的`generate_media`工具
        4. 提供简短的解释，说明所做的创意选择
        5. 如果请求不清晰，询问关于风格偏好的澄清

        视频将在UI中自动显示在您的回复下方。
        始终旨在创建引人入胜且有意义的视频，使用户的愿景变为现实！\
    """),
    markdown=True,
    show_tool_calls=True,
)

# 使用示例
video_agent.print_response(
    "生成一段穿越色彩斑斓星云的宇宙旅程", stream=True
)

# 检索并显示生成的视频
videos = video_agent.get_videos()
if videos:
    for video in videos:
        print(f"生成的视频URL: {video.url}")

# 更多要尝试的示例提示：
"""
尝试这些创意提示：
1. "创建一个秋叶在宁静森林中飘落的视频"
2. "生成一个猫玩球的视频"
3. "创建一个宁静的锦鲤池塘和涟漪的视频"
4. "生成一个温馨壁炉和跳动的火焰的视频"
5. "创建一个神秘传送门在魔法境界中开启的视频"
"""
