"""🎤 使用GPT-4的音频输入/输出 - 使用Agno创建语音交互

此示例展示了如何创建一个可以处理音频输入并生成音频响应的AI代理。
您可以将此代理用于各种基于语音的交互，从分析语音内容到生成自然声音的响应。

要尝试的音频交互示例：
- 上传对话录音进行分析
- 让代理用语音回答问题
- 处理不同的语言和口音
- 分析语音中的语气和情感

运行 `pip install openai requests agno` 安装依赖。
"""

from textwrap import dedent

import requests
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from agno.agent import Agent
from agno.media import Audio
from agno.models.openai import OpenAIChat
from agno.utils.audio import write_audio_to_file


# 创建AI语音交互代理
agent = Agent(
    model=OpenAIChat(
        id="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "sage", "format": "wav"},
    ),
    description=dedent("""\
        您是音频处理和语音交互专家，能够理解和分析口语内容，
        同时提供自然、引人入胜的语音响应。
        您擅长理解语音中的上下文、情感和细微差别。\
    """),
    instructions=dedent("""\
        作为语音交互专家，请遵循这些指南：
        1. 仔细聆听音频输入，以理解内容和上下文
        2. 提供简明扼要的回答，解决主要问题
        3. 生成语音回复时，保持自然、对话式的语调
        4. 在分析中考虑说话者的语调和情感
        5. 如果音频不清晰，请寻求澄清

        专注于创建引人入胜且有帮助的语音交互！\
    """),
)

# 获取音频文件并将其转换为base64编码字符串
url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()

# 处理音频并获取响应
agent.run(
    "这段录音里有什么？请分析内容和语调。",
    audio=[Audio(content=response.content, format="wav")],
)

# 如果有音频响应，保存它
if agent.run_response.response_audio is not None:
    write_audio_to_file(
        audio=agent.run_response.response_audio.content, filename="tmp/response.wav"
    )

# 更多要尝试的交互示例：
"""
尝试这些语音交互场景：
1. "您能总结一下这段录音中讨论的要点吗？"
2. "您在说话者的声音中检测到什么情绪或语调？"
3. "请提供关于语音模式和清晰度的详细分析"
4. "您能识别任何背景噪音或音频质量问题吗？"
5. "这段录音的整体背景和目的是什么？"

注意：您可以通过将自己的音频文件转换为base64格式来使用它们。
使用自己的音频文件的示例：

with open('your_audio.wav', 'rb') as audio_file:
    audio_data = audio_file.read()
    agent.run("分析这段音频", audio=[Audio(content=audio_data, format="wav")])
"""
