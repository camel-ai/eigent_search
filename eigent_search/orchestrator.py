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

from __future__ import annotations
import asyncio

from camel.logger import get_logger
from camel.responses import ChatAgentResponse
import nest_asyncio
from pydantic import BaseModel
import tenacity

from .config import SearchConfig
from .tool_trajectory import ToolTrajectory
from .toolkit import QueryProcessingToolkit

logger = get_logger(__name__)


class SearchResponse(BaseModel):
    response: ChatAgentResponse
    tool_trajectory: list[ToolTrajectory]

    @property
    def token_usage(self) -> int:
        return self.response.info["usage"]["total_tokens"]


class ErrorSearchResponse(BaseModel):
    error: str


class SearchOrchestrator:
    """Orchestrates interaction between `SearchAgent` and `SearchEnvironment`.

    Responsibilities:
    - Coordinates agent-environment interaction
    - Manages retry logic with exponential backoff
    - Handles timeouts and error recovery
    - Tracks metrics (tool trajectories, token usage)
    - Manages lifecycle (reset, cleanup)
    """

    def __init__(self, config: SearchConfig):
        """Initialize the orchestrator.

        Args:
            agent: The DeepSearchAgent instance
            environment: The DeepSearchEnvironment instance
            max_retries: Maximum number of retry attempts (default: 5)
            timeout_minutes: Timeout in minutes for each step (default: 5)
        """
        self.config = config
        self.agent = config.create_agent()
        self.query_processing_toolkit = None
        for toolkit in config.toolkits:
            if isinstance(toolkit, QueryProcessingToolkit):
                self.query_processing_toolkit = toolkit

    async def cleanup(self):
        """Clean up resources."""
        # NOTE: Possible to be extended to parallel cleanup in the future but not necessary for now
        for toolkit in self.config.toolkits_to_cleanup:
            try:
                await toolkit.cleanup()
            except Exception as e:
                logger.warning(f"Error during cleanup of {type(toolkit).__name__}: {e}")

    def reset(self):
        """Reset the agent and cleanup resources."""
        self.agent.reset()
        nest_asyncio.apply()
        asyncio.run(self.cleanup())

    async def astep(self, input_query: str) -> ChatAgentResponse:
        if self.query_processing_toolkit:
            self.query_processing_toolkit.reset(input_query)
        response = await self.agent.astep(
            input_query,
            response_format=self.config.response_format,
        )
        return response

    def run_agent(self, input_query: str) -> SearchResponse:
        """Run the agent with retry logic, timeout and error handling.

        Args:
            input_query: The user's search query

        Returns:
            SearchResponse containing the agent's response and metadata

        Raises:
            Exception: If execution fails or times out after all retries
        """

        def _handle_retry(retry_state: tenacity.RetryCallState):
            logger.error(
                f"Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}. Reset agent, cleanup resources and retry."
            )
            self.reset()

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self.config.max_orchestrator_retries),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
            before_sleep=_handle_retry,
        )
        def _run_with_retry():
            nest_asyncio.apply()
            timeout_seconds = self.config.timeout_minutes_per_orchestrator_step * 60
            response = asyncio.run(
                asyncio.wait_for(
                    self.astep(input_query),
                    timeout=timeout_seconds,
                )
            )
            return SearchResponse(
                response=response,
                tool_trajectory=ToolTrajectory.extract_from_response(response),
            )

        try:
            return _run_with_retry()
        except Exception as e:
            logger.error(
                f"Agent failed after {self.config.max_orchestrator_retries} attempts: {e}"
            )
            self.reset()
            return ErrorSearchResponse(error=str(e))
