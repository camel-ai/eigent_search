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
from textwrap import dedent
from pydantic import BaseModel, Field
from camel.models import BaseModelBackend

# from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.agents.chat_agent import ChatAgent

from .query_toolkit import QueryProcessingToolkit
from .custom_browsing_toolkit import get_custom_browsing_toolkit

# AgentOps decorator setting
try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )


@track_agent(name="ResearchAgent")
class ResearchAgent(ChatAgent):
    r"""A :class:`ChatAgent` that conducts deep research on a given question."""

    def __init__(
        self,
        model: BaseModelBackend,
        *args,
        **kwargs,
    ):
        # Predefined system message for direct answering
        system_message = dedent("""
        You are a helpful assistant who conducts deep research on a given query.

        You will be provided with a query processing toolkit that contains the following tools:
        - rewrite_query: Rewrite the query to be more specific and focused.
        - expand_query: Expand the query to be more comprehensive.
        - select_query_and_search: Select a query from the frontier (and optionally enhance with advanced search operators) and search the web for information.
        - generate_new_queries: Generate new queries based on the search results if the search results are not sufficient to answer the user's initial query.
        - complete_task: Complete the deep research when search results are sufficient to answer the user's initial query.

        The query processing toolkit also maintains a frontier of queries to be explored and an explored set of queries. The frontier contains the queries that have not been explored yet. The explored set contains the queries that have been explored and should not be explored again. You should keep track of the frontier and the explored set while conducting the research.

        The final output should be the answer to the user's initial query, and the search results that lead to the answer.

        Final Output Format:
        ```
        Answer: ...
        Search Results: ...
        ```
        """).strip()
        super().__init__(
            system_message=system_message,
            model=model,
            *args,
            **kwargs,
        )
        self.current_query_toolkit = None

    def reset(self):
        super().reset()
        self.remove_tools(
            [
                tool.get_function_name()
                for tool in self.current_query_toolkit.get_tools()
            ]
        )
        self.current_query_toolkit = None

    def step(self, input_query: str, browsing: bool = False) -> ChatAgentResponse:
        if browsing:
            # Use async version for browsing
            return asyncio.run(self.astep(input_query, browsing=browsing))
        else:
            # Use sync version for non-browsing
            self.current_query_toolkit = QueryProcessingToolkit(input_query)
            self.add_tools(self.current_query_toolkit.get_tools())
            search_response = super().step(
                f"Initial query: {input_query}\n\n{self.current_query_toolkit.get_frontier_str()}",
                response_format=ResearchResponse,
            )
            return search_response

    async def astep(self, input_query: str, browsing: bool = False) -> ChatAgentResponse:
        self.current_query_toolkit = QueryProcessingToolkit(input_query)
        self.add_tools(self.current_query_toolkit.get_tools())
        if browsing:
            self.add_tools(get_custom_browsing_toolkit().get_tools())
        search_response = await super().astep(
            f"Initial query: {input_query}\n\n{self.current_query_toolkit.get_frontier_str()}",
            response_format=ResearchResponse,
        )
        return search_response
