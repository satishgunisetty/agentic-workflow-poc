from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools.weather_tools import get_weather_alert_by_code
from agents.base_agent import BaseAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    """
    Weather Agent that provides weather information based on city/state codes.
    Inherits from BaseAgent.
    """

    def __init__(
        self, llm: BaseChatModel, tools: List[BaseTool] = [get_weather_alert_by_code]
    ):
        """
        Initialize the WeatherAgent with a language model and tools.
        Args:
            llm (BaseChatModel): The language model to use.
            tools (List[BaseTool]): List of tools to be used by the agent. If None,
            defaults to using get_weather_alert_by_code.
        """

        super().__init__(llm, tools=tools)

        self.tool_names = [tool.get_name() for tool in self.tools]

        self._init_agent_prompt()

        self._init_agent()

    def _init_agent_prompt(self):
        """
        Initialize the agent prompt for the WeatherAgent.
        """

        system_prompt = f"""
        You are a weather assistant that provides weather alerts for specific US states
        using the get_weather_alert_by_code tool.

        For every user query:
        1. Identify any US state name or 2-letter code in the input.
        2. If a state name is provided (e.g., California), convert it to
            its UPPERCASE 2-letter code (e.g., CA).
        3. Call the appropriate tool
        4. Return the tool's output or an appropriate error message
            if no state is identified or the tool fails.

        Example:
        User: "What is the weather alert for California?"
        Assistant: Let me check the weather alerts for CA.
            [Calls get_weather_alert_by_code with "CA"]

        Available tools: {self.tool_names}
        """

        self.agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

    def _init_agent(self):
        """
        Initialize the agent with the prompt and tools.
        This method should be called in the subclass constructor.
        """

        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt,
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
        )

    async def execute(
        self, query: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query to get weather alerts for a specific state code.

        Args:
            query (str): User input query.
            chat_history (List[Dict[str, Any]]): List of previous messages
                in the conversation.

        Returns:
            Dict[str, Any]: The response from the agent after processing the query.
        """

        try:
            if not isinstance(chat_history, list):
                logger.info("chat_history must be a list; defaulting to empty list")
                chat_history = []

            # Ensure query is properly formatted
            formatted_query = query.strip()

            logger.info(f"Executing query: {formatted_query}")

            if not formatted_query:
                return {"error": "Empty query provided"}

            result = await self.agent_executor.ainvoke(
                {
                    "input": formatted_query,
                    "chat_history": chat_history,
                    "agent_scratchpad": [],
                }
            )
            return result
        except Exception as e:
            logger.error(f"Failed to execute  query '{query}': {str(e)}")
            return {"error": f"Unable to process the  query: {str(e)}"}
