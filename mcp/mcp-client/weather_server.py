import os

import httpx
from dotenv import load_dotenv
from httpx import HTTPStatusError

load_dotenv()
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")
USER_AGENT = "weather-app/1.0"

@mcp.tool()
async def get_weather(city: str):
    """
    获取指定城市当前的天气情况
    :param city: 必要参数，城市名，应当为英文形式
    :return: 接口响应，json格式
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    params = {
        "q": city,
        "appid": os.getenv("OWM_APPID"),
        "units": "metric",
        "lang": "zh_cn"
    }
    # response = requests.get(url, params=params)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except HTTPStatusError as e:
        return {"error": f"HTTP Error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Unknown Error: {e}"}


if __name__ == "__main__":
    # mcp.run(transport="stdio")
    mcp.run(transport="sse")