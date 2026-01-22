import os
import json
import httpx
import requests
from typing import Any
from dotenv import load_dotenv
load_dotenv()
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")


@mcp.tool()
async def get_weather(city: str):
    """
    获取指定城市当前的天气情况
    :param city: 必要参数，城市名，应当为英文形式
    :return: 接口响应，json格式
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OWM_APPID"),
        "units": "metric",
        "lang": "zh_cn"
    }
    response = requests.get(url, params=params)
    return response.json()

if __name__ == "__main__":
    mcp.run(transport="stdio")