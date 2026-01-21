from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def summarize_website(url: str) -> str:
    """访问指定网站并返回内容总结"""
    try:
        sync_browser = create_sync_playwright_browser()
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
        tools = toolkit.get_tools()
        prompt = hub.pull("hwchase17/openai-tools-agent")
        model = init_chat_model("deepseek-chat", model_provider="deepseek")
        agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        command = {
        "input": f"访问这个网站 {url} 并详细总结一下这个网站的内容。"
        }
        response = agent_executor.invoke(command)
        return response.get("output", "无法获取网站内容总结。")
    except Exception as e:
        return f"网站访问失败：{e}"

@tool
def generate_pdf(text: str) -> str:
    """将输入内容转换为pdf保存在本地"""
    return "PDF文件已成功生成"

simple_chain = summarize_website | generate_pdf

from langchain_classic.tools import Tool

# 输入和输出都是str，通过args_schema实现更复杂的输入控制
simple_tool = Tool.from_function(
    func=simple_chain.invoke,
    description="访问指定网站，并进行总结，将结果保存为pdf文件。",
    name="summary_and_save"
)