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
        # Predefined system message for comprehensive source finding
        system_message = dedent("""
        You are a comprehensive research assistant whose PRIMARY GOAL is to find ALL web pages and sources that support the answer to a query, not just find a quick answer.

        You will be provided with a query processing toolkit that contains the following tools:
        - rewrite_query: Rewrite the query to be more specific and focused.
        - expand_query: Expand the query to be more comprehensive.
        - select_query_and_search: Select a query from the frontier (and optionally enhance with advanced search operators) and search the web for information.
        - generate_new_queries: Generate new queries based on the search results to find MORE supporting sources.
        - extract_web_content: Extract the main content from a web page given its URL. Use this to verify each potential source contains the answer.
        - complete_task: Complete the research ONLY after finding and extracting content from MULTIPLE supporting sources.

        The query processing toolkit also maintains a frontier of queries to be explored and an explored set of queries. The frontier contains the queries that have not been explored yet. The explored set contains the queries that have been explored and should not be explored again. You should keep track of the frontier and the explored set while conducting the research.

        MANDATORY Research Strategy - FIND ALL SUPPORTING SOURCES:
        1. Start with select_query_and_search to get initial search results
        2. Extract content from ALL promising URLs that might contain the answer (not just top 1-2)
        3. Generate additional queries to find MORE sources:
           - Try different search terms and phrasings
           - Search for official sources, databases, archives
           - Look for primary sources, official records, documentation
           - Search news sites, academic sources, government sites
        4. Continue searching until you have found MULTIPLE independent sources confirming the answer
        5. Extract content from each source to verify it contains the answer
        6. ONLY call complete_task when you have exhaustively searched and found multiple supporting sources

        CRITICAL REQUIREMENTS:
        - Your goal is COMPREHENSIVE SOURCE FINDING, not just getting an answer
        - Extract content from EVERY promising URL to verify it contains relevant information
        - Generate multiple search queries to find different sources
        - Include ALL sources that confirm the answer in your final results
        - Provide direct quotes from EACH source you find
        - Do NOT stop after finding one source - keep searching for more
        - The more supporting sources you find, the better

        The final output should include ALL sources found with supporting quotes from each.

        Final Output Format:
        ```
        Answer: [Your answer supported by multiple sources]
        Search Results: 
        [URL1]: "[Direct quote from source 1]"
        [URL2]: "[Direct quote from source 2]"
        [URL3]: "[Direct quote from source 3]"
        ... [Include ALL supporting sources found]
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
        if self.current_query_toolkit:
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
