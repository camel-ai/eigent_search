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

from textwrap import dedent
from pydantic import BaseModel, Field
from camel.models import BaseModelBackend

# from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.agents.chat_agent import ChatAgent

from .query_toolkit import QueryProcessingToolkit

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
        You are a helpful assistant who conducts deep, systematic research to answer a given query.
                                
        You have access to the following query processing tools:
        1. rewrite_query(query, rewritten_query)
            - Purpose: Improve the clarity, specificity, or focus of a vague or ambiguous query in the frontier.
            - Primary use: Applied mainly to the **initial user query** before the main search process begins.
            - Use when: The current query is too broad, unclear, or imprecise.
        2. decompose_query(query, expanded_queries)
            - Purpose: Break a complex or multi-faceted query into simpler, more focused sub-queries.
            - Primary use: Applied mainly to the **initial user query** to identify separate aspects or subtopics for targeted searching.
            - Use when: The current query is too complex or multifaceted, and needs to be simplified or split into multiple parts.
        3. select_query_and_search(query, enhanced_query)
            - Purpose: Select the most promising query from the frontier and perform a web search.
            - Use when: There is at least one query in the frontier and it is ready to retrieve new information for the most valuable frontier query.
        4. generate_new_queries(search_results, new_queries)
            - Purpose: Create new queries based on gaps, leads, or new angles discovered in the latest search results.
            - Use when: Search results are insufficient or reveal promising new directions. 
        5. reflect(reflction)
            - Purpose: Assess current coverage and decide the next steps.
            - Use often to guide the research process and avoid wasted searches.
        6. complete_task(search_results, final_answer)
            - Purpose: Finalize the research when sufficient, credible evidence exists to answer the initial query.
                                                       
        General Instructions:
        - Always keep track of:
            • The frontier (queries to be explored)
            • The explored set (queries already searched)
            • The search results collected (URLs, description, etc.)
        - Avoid repeating searches for the same or equivalent queries.
        - Be efficient — minimize the number of searches while maximizing coverage.
        - Ensure all claims in the final answer are supported by credible, relevant sources.
        - Terminate promptly when sufficient evidence is gathered.
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

    def step(self, input_query: str) -> ChatAgentResponse:
        self.current_query_toolkit = QueryProcessingToolkit(input_query)
        self.add_tools(self.current_query_toolkit.get_tools())
        search_response = super().step(
            f"Initial query: {input_query}\n\n{self.current_query_toolkit.get_frontier_str()}",
            response_format=ResearchResponse,
        )
        return search_response
