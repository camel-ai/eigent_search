# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========

from typing import Any, Dict, List, Type

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
from camel.responses import ChatAgentResponse
from pydantic import BaseModel
import tenacity

from eigent_search.research.tool_trajectory import ToolTrajectory

logger = get_logger(__name__)


def extract_token_usage(response: ChatAgentResponse) -> int:
    """Extract token usage from ChatAgentResponse.

    Args:
        response: The ChatAgentResponse object

    Returns:
        Total token usage as an integer
    """
    return response.info["usage"]["total_tokens"]


def run_agent_with_retry(
    agent: ChatAgent,
    input_query: str,
    response_format: Type[BaseModel],
    max_retries: int = 5,
    timeout_minutes: int = 5,
) -> dict:
    """Run agent.step with exponential retry logic.

    Args:
        agent: The agent to run
        problem: The problem text
        agent_type: Type of agent for logging
        hash_id: Hash ID for logging
        max_retries: Maximum number of retry attempts (default 5)

    Returns:
        Parsed response dict from agent

    Raises:
        Exception: If all retries fail
    """

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}. Retrying..."
        ),
        reraise=False,
    )
    def _run_with_retry(input_query: str) -> tuple[dict, List[Dict[str, Any]]]:
        import asyncio
        import nest_asyncio

        nest_asyncio.apply()
        response = asyncio.run(
            asyncio.wait_for(
                agent.astep(input_query, response_format=response_format),
                timeout=timeout_minutes * 60,
            )
        )
        # Extract tool trajectory and token usage
        tool_trajectory = ToolTrajectory.extract_from_response(response)
        token_usage = extract_token_usage(response)
        logger.info(f"[Tool Trajectory]:\n{tool_trajectory.model_dump_json(indent=2)}")
        logger.info(f"[Token Usage]: {token_usage}")

        return {
            "response": eval(response.msgs[0].content),
            "tool_trajectory": tool_trajectory.model_dump(),
            "token_usage": token_usage,
        }

    try:
        result = _run_with_retry(input_query)
        return result

    except Exception:
        # If all retries fail, return a dummy result, with error flag
        logger.error(f"All {max_retries} attempts failed for query: {input_query}")
        return {
            "response": {
                "error": True,
            },
            "tool_trajectory": [],
        }
