from openai import OpenAI
from transformers import Qwen3Model


def test_chat():
    openai_api_key = "EMPTY"
    openai_api_base = "http://b4b2c947e7a149cc8df102745564fd03.qhdcloud.lanyun.net:12500/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )

    messages = [
        {
            "role": "user",
            "content": "你好，好久不见！"
        }
    ]

    response = client.chat.completions.create(
        model="Qwen3-8B-unsloth-bnb-4bit",
        messages=messages
    )

    print(response.choices[0].message.content)