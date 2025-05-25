from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        """
        Initialize the BaseAgent with a language model and tool names.
        Args:
            llm (BaseChatModel): The language model to use.
            tool_names (list[BaseTool]): List of tools to be used by the agent.
        """
        if not isinstance(llm, BaseChatModel):
            raise TypeError("llm must be an instance of BaseChatModel.")

        if not isinstance(tools, list) or not all(
            isinstance(tool, BaseTool) for tool in tools
        ):
            raise TypeError("tools must be a list of BaseTool instances.")

        self.llm = llm
        self.tools = tools

        logger.info("BaseAgent initialized with %d tools", len(tools))

        self.agent_prompt = None

    @abstractmethod
    def _init_agent_prompt(self) -> str:
        """
        Abstract method to get the agent prompt.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _init_agent(self):
        """
        Initialize the agent with the prompt.
        This method should be called in the subclass constructor.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def execute(
        self, query: str, chat_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query. To be implemented by subclasses.

        Args:
            query: User input query
            chat_history: List of previous messages in the conversation

        Returns:
            Response Dict[str, Any]: Processed response from the agent.
        """
        raise NotImplementedError("Subclasses must implement process_query")
