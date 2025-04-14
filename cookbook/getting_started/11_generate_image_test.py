"""🎨 使用DALL-E生成图像 - 使用Agno创建AI艺术

此示例展示了如何创建一个使用DALL-E生成图像的AI代理。
您可以使用此代理创建各种类型的图像，从逼真的照片到艺术插图和创意概念。

要尝试的示例提示：
- "创建一幅在日落时云中漂浮的城市的超现实画作"
- "生成一张舒适咖啡店内部的照片写实图像"
- "为科技创业公司设计一个可爱的卡通吉祥物"
- "创建一幅赛博朋克武士的艺术肖像"

运行 `pip install openai agno` 安装依赖。
"""

from textwrap import dedent

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from agno.agent import Agent
from agno.models.openai import QwenOmniTurboChat, Qwen2Dot5Omni7bChat
from agno.tools.dalle import DalleTools


# 创建创意AI艺术家代理
image_agent = Agent(
    model=Qwen2Dot5Omni7bChat(),
    tools=[DalleTools()],
    description=dedent("""\
        您是一位经验丰富的AI艺术家，擅长各种艺术风格，
        从照片写实主义到抽象艺术。您对构图、色彩理论和视觉讲故事有深刻理解。\
    """),
    instructions=dedent("""\
        作为AI艺术家，请遵循以下指南：
        1. 仔细分析用户的请求，了解所需的风格和情绪
        2. 在生成之前，用艺术细节如光线、视角和氛围丰富提示
        3. 使用带有详细、精心制作的提示的`create_image`工具
        4. 提供简短的解释，说明所做的艺术选择
        5. 如果请求不清晰，询问关于风格偏好的澄清

        始终旨在创建视觉冲击力强且有意义的图像，捕捉用户的愿景！\
    """),
    markdown=True,
    show_tool_calls=True,
    stream=True,
    use_json_mode=True,
)

# 使用示例
image_agent.print_response(
    "创建一个超现实主义有漂浮书籍和发光水晶的魔法图书馆", stream=True
)

# 检索并显示生成的图像
images = image_agent.get_images()
if images and isinstance(images, list):
    for image_response in images:
        image_url = image_response.url
        print(f"生成的图像URL: {image_url}")

# 更多要尝试的示例提示：
"""
尝试这些创意提示：
1. "生成一个蒸汽朋克风格的机器人演奏小提琴"
2. "设计一个樱花季节的宁静禅宗花园"
3. "创建一个带有生物发光建筑的水下城市"
4. "生成一个雪夜中温馨的小木屋"
5. "创建一个有飞行汽车和摩天大楼的未来城市景观"
"""
