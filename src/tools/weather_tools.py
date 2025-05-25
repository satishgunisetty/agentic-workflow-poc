from langchain_core.tools import tool
from typing import Dict, Any
import httpx
import logging

from utils.constants import WEATHER_API_BASE, USER_AGENT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_alert(alert: Dict[str, Any]) -> str:
    """Format a weather alert into a string."""
    props = alert.get("properties", {})
    if not props:
        return ""

    return f"""
        Event: {props.get("event", "Unknown")}
        Description: {props.get("description", "No description available")}
        Severity: {props.get("severity", "Unknown")}
        Area: {props.get("areaDesc", "Unknown")}
        Instructions: {props.get("instruction", "No instructions available")}
    """


@tool
async def get_weather_alert_by_code(code: str) -> str | None:
    """
    Get weather information by city/state code.

    Args:
        code (str): The city/state code to get the weather for.
            Eg: "CA" for California, "NY" for New York.

    Returns:
        str: Weather information for the specified city code.
    """
    # Placeholder implementation
    logger.info(f"Fetching weather for city code: {code}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }

    url = f"{WEATHER_API_BASE}/alerts/active/area/{code}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url=url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            result = response.json()

            if not result or "features" not in result:
                logger.info(
                    "No features found in the response or unable to fetch alerts."
                )
                return "Unable to fetch alerts or No alerts found for the given state."

            if not result.get("features"):
                logger.info("No alerts found for the given state.")
                return "No alerts found for the given state."

            alerts = [format_alert(feature) for feature in result["features"]]

            if not alerts:
                logger.info("No alerts found for the given state.")
                return "No alerts found for the given state."

            return "\n---\n".join(alerts)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None
