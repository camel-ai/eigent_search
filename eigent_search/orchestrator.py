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
from typing import Type

from camel.logger import get_logger
from camel.responses import ChatAgentResponse
import nest_asyncio
from pydantic import BaseModel
import tenacity

from .config import SearchConfig
from .tool_trajectory import ToolTrajectory
from .toolkit import EigentSearchToolkit, QueryProcessingToolkit

logger = get_logger(__name__)


class SearchRequest(BaseModel):
    query_id: str
    input_query: str


class SearchResult(SearchRequest):
    response: ChatAgentResponse
    response_format: Type[BaseModel] | None
    tool_trajectory: ToolTrajectory

    @property
    def token_usage(self) -> int:
        return self.response.info["usage"]["total_tokens"]

    @property
    def formatted_response(self) -> str:
        response = self.response.msgs[0].content.strip()
        if self.response_format:
            return self.response_format.model_validate_json(response).model_dump_json(
                indent=2
            )
        return response


class ErrorSearchResult(SearchRequest):
    error: str


class SearchOrchestrator:
    """Orchestrates search agent execution."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.agent = config.create_agent()
        self.eigent_search_toolkit = None
        self.query_processing_toolkit = None
        for toolkit in config.toolkits:
            if isinstance(toolkit, EigentSearchToolkit):
                self.eigent_search_toolkit = toolkit
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

    async def astep(
        self, input_query: str, query_id: str | None = None
    ) -> ChatAgentResponse:
        # Load initial query into query processing toolkit
        if self.query_processing_toolkit:
            self.query_processing_toolkit.load_initial_query(input_query)

        # Update note-taking directory for eigent search toolkit
        if self.eigent_search_toolkit:
            note_taking_directory = self.config.working_directory / "note_taking_logs"
            if query_id:
                note_taking_directory = note_taking_directory / query_id
            self.eigent_search_toolkit.update_note_taking_directory(
                note_taking_directory
            )

        response = await self.agent.astep(
            input_query,
            response_format=self.config.response_format,
        )
        return response

    def create_search_request(
        self, input_query: str, query_id: str | None = None
    ) -> SearchRequest:
        return SearchRequest(input_query=input_query, query_id=query_id)

    def run_agent(self, search_input: SearchRequest) -> SearchResult | ErrorSearchResult:
        """Run the agent with retry logic, timeout and error handling."""

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
                    self.astep(search_input.input_query, search_input.query_id),
                    timeout=timeout_seconds,
                )
            )
            return SearchResult(
                **search_input.model_dump(),
                response=response,
                response_format=self.config.response_format,
                tool_trajectory=ToolTrajectory.extract_from_response(response),
            )

        try:
            result = _run_with_retry()
            self.reset()
            return result
        except Exception as e:
            logger.error(
                f"Agent failed after {self.config.max_orchestrator_retries} attempts: {e}"
            )
            self.reset()
            return ErrorSearchResult(error=str(e))
        finally:
            self.reset()
