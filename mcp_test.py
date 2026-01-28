# mcp client
# mcp server

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
chat_model = init_chat_model("deepseek-chat", model_provider="deepseek")

import json
mcp_server_cfg_path = "mcp_server_cfg.json"
with open(mcp_server_cfg_path, "r") as fp:
    mcp_server_cfg = json.load(fp)["mcpServers"]
print(mcp_server_cfg)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

mcp_client = MultiServerMCPClient(mcp_server_cfg)

import asyncio
async def get_mcp_tools():
    tools = await mcp_client.get_tools()
    return tools

asyncio.run(get_mcp_tools())
