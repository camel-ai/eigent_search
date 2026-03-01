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
import traceback
from typing import Type

from camel.logger import get_logger
from camel.responses import ChatAgentResponse
import nest_asyncio
from openai import BadRequestError
from pydantic import BaseModel
import tenacity

# Import minimax patch to ensure it's applied before tool_trajectory is imported
# The patch auto-applies on import (see minimax_m25_patch.py line 486)
try:
    import eigent_search.minimax_m25_patch  # noqa: F401
except ImportError:
    pass  # Minimax patch module not available

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
        try:
            response = self.response.msgs[0].content.strip()
        except IndexError:
            logger.error(f"Empty response: {self.response}")
            return ""

        # CRITICAL: For Minimax M2.5, strip <think> tags from the response
        # The reasoning content will be available in reasoning_content property
        if '<think>' in response:
            import re
            # Remove all <think>...</think> blocks
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = response.strip()

        if self.response_format:
            return self.response_format.model_validate_json(response).model_dump_json(
                indent=2
            )
        return response

    @property
    def reasoning_content(self) -> str | None:
        """Extract reasoning_content from the response.

        Supports:
        - DeepSeek R1: reasoning_content field
        - Minimax M2.5: <think> tags in content
        """
        try:
            # Try to get reasoning_content from the first message
            if self.response.msgs and len(self.response.msgs) > 0:
                msg = self.response.msgs[0]

                # DeepSeek R1 style: reasoning_content attribute
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    return msg.reasoning_content

                # Minimax M2.5 style: extract <think> tags from content
                if hasattr(msg, 'content') and msg.content and '<think>' in msg.content:
                    import re
                    pattern = r'<think>(.*?)</think>'
                    matches = re.findall(pattern, msg.content, re.DOTALL | re.IGNORECASE)
                    if matches:
                        # Join all <think> blocks with newlines
                        return '\n'.join(match.strip() for match in matches)

            # If not in msgs, try to get from response.info (raw API response)
            if hasattr(self.response, 'info') and isinstance(self.response.info, dict):
                if 'choices' in self.response.info and len(self.response.info['choices']) > 0:
                    message = self.response.info['choices'][0].get('message', {})
                    if 'reasoning_content' in message:
                        return message['reasoning_content']

            return None
        except Exception as e:
            logger.warning(f"Failed to extract reasoning_content: {e}")
            return None


class ErrorSearchResult(SearchRequest):
    error: str


class GoogleAPILimitError(Exception):
    """Raised when Google Search API daily limit is reached."""

    pass


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

    async def astep(self, input_query: str) -> ChatAgentResponse:
        # Load initial query into query processing toolkit
        if self.query_processing_toolkit:
            input_query = self.query_processing_toolkit.load_initial_query(input_query)

        response = await self.agent.astep(
            input_query,
            response_format=self.config.response_format,
        )
        return response

    def create_search_request(
        self, input_query: str, query_id: str | None = None
    ) -> SearchRequest:
        return SearchRequest(input_query=input_query, query_id=query_id)

    def run_agent(
        self, search_input: SearchRequest
    ) -> SearchResult | ErrorSearchResult:
        """Run the agent with retry logic, timeout and error handling."""

        def is_content_filter_error(exc: BaseException) -> bool:
            """Check if exception is an Azure content filter error."""
            if isinstance(exc, BadRequestError):
                error_body = getattr(exc, "body", {}) or {}
                error_info = (
                    error_body.get("error", {}) if isinstance(error_body, dict) else {}
                )
                return error_info.get("code") == "content_filter"
            return False

        def is_google_api_limit_error(exc: BaseException) -> bool:
            """Check if exception is a Google API daily limit error."""
            return isinstance(exc, GoogleAPILimitError)

        def should_retry(exc: BaseException) -> bool:
            """Retry all exceptions except content filter and Google API limit errors."""
            if is_content_filter_error(exc):
                logger.warning(
                    f"[{search_input.query_id}] Content filter triggered, not retrying"
                )
                return False
            if is_google_api_limit_error(exc):
                logger.error(
                    f"[{search_input.query_id}] Google API daily limit reached, "
                    "interrupting benchmark"
                )
                return False
            return True

        def on_retry(retry_state: tenacity.RetryCallState):
            """Log and reset agent before each retry."""
            exc = retry_state.outcome.exception()
            logger.error(
                f"[{search_input.query_id}] Attempt {retry_state.attempt_number} failed: "
                f"{type(exc).__name__}: {exc!r}\n"
                f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            self.reset()

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(self.config.max_orchestrator_retries),
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            retry=tenacity.retry_if_exception(should_retry),
            reraise=True,
            before_sleep=on_retry,
        )
        def run_with_retry() -> SearchResult:
            nest_asyncio.apply()
            timeout_seconds = self.config.timeout_minutes_per_orchestrator_step * 60
            response = asyncio.run(
                asyncio.wait_for(
                    self.astep(search_input.input_query),
                    timeout=timeout_seconds,
                )
            )
            tool_trajectory = ToolTrajectory.extract_from_response(response)

            # Check for Google API daily limit error in tool results
            if tool_trajectory.has_google_api_limit_error():
                raise GoogleAPILimitError(
                    "Google Search API daily limit reached. Benchmark must be interrupted."
                )

            return SearchResult(
                **search_input.model_dump(),
                response=response,
                response_format=self.config.response_format,
                tool_trajectory=tool_trajectory,
            )

        try:
            result = run_with_retry()
            return result
        except GoogleAPILimitError:
            # Re-raise to interrupt the benchmark
            raise
        except Exception as e:
            if is_content_filter_error(e):
                logger.warning(f"[{search_input.query_id}] Skipped due to content filter")
            else:
                logger.error(
                    f"[{search_input.query_id}] Failed after "
                    f"{self.config.max_orchestrator_retries} attempts: {e}"
                )
            return ErrorSearchResult(**search_input.model_dump(), error=str(e))
        finally:
            logger.info(f"[{search_input.query_id}] Cleaning up resources...")
            self.reset()
