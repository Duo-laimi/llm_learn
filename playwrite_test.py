from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_openai_functions_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")

sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

prompt = hub.pull("hwchase17/openai-tools-agent")
print(prompt)

model = init_chat_model("deepseek-chat", model_provider="deepseek")

agent = create_openai_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

command = {
    "input": "访问这个网站https://www.zhihu.com/hot，简要介绍前3条热点信息。"
}

response = agent_executor.invoke(command)

print(response)