import logging
import asyncio
from dotenv import load_dotenv
from os import environ

from langchain_openai import AzureChatOpenAI

from agents.weather_agent import WeatherAgent

from tools.weather_tools import get_weather_alert_by_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    print("Hello from agentic-workflow-poc!")

    load_dotenv()

    azure_deployment = environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = environ.get("AZURE_OPENAI_API_VERSION")

    llm = AzureChatOpenAI(
        azure_deployment=azure_deployment,
        api_version=api_version,
    )

    agent = WeatherAgent(
        llm=llm,
    )

    chat_history = []

    response = await agent.execute(
        query="What is the weather alert for California?", chat_history=chat_history
    )

    print("Response:", response)


async def func():

    result = await get_weather_alert_by_code("CA")
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())

    # asyncio.run(func())
