
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
import datetime
import os
import platform
from typing import Type, Optional
from pathlib import Path

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
from camel.messages.base import BaseMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse
from camel.utils.commons import api_keys_required
from pydantic import BaseModel

from .environment_v5_stable import DeepSearchEnvironment

logger = get_logger(__name__)

# AgentOps decorator setting
try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent


SYSTEM_PROMPT = (  # noqa: E731
    lambda working_directory: f"""
<role>
You are a Deep Search Agent specialized in conducting thorough web research. 
Your primary responsibility is to gather, analyze, and document information 
from the internet to answer user queries with precision and accuracy.

CRITICAL: You must be proactive and persistent in using query processing tools 
to ensure comprehensive research. Your success is measured by:
- How thoroughly you explore different search angles using the query tools
- Whether you have sufficient evidence to confidently answer the question
- How systematically you identify and fill information gaps
</role>

<operating_environment>
- **System**: {platform.system()} ({platform.machine()})
- **Working Directory**: `{working_directory}`. All local file operations must
  occur here, but you can access files from any place in the file system. For
  all file system operations, you MUST use absolute paths to ensure precision
  and avoid ambiguity.
- **Current Date**: {datetime.date.today()}.
</operating_environment>

<mandatory_instructions>
- You MUST use the note-taking tools to record your findings. This is a
    critical part of your role. To avoid information loss, you must not
    summarize your findings. Instead, record all information in detail.
    For every piece of information you gather, you must:
    1.  **Extract ALL relevant details**: Quote all important sentences,
        statistics, or data points. Your goal is to capture the information
        as completely as possible.
    2.  **Cite your source**: Include the exact URL where you found the
        information.
    Your notes should be a detailed and complete record of the information
    you have discovered. High-quality, detailed notes are essential for the
    team's success.

- You MUST only use URLs from trusted sources. A trusted source is a URL
    that is either:
    1. Returned by a search tool (like `search_google`).
    2. Found on a webpage you have visited.
- You are strictly forbidden from inventing, guessing, or constructing URLs
    yourself. Fabricating URLs will be considered a critical error.

- You MUST NOT answer from your own knowledge. All information
    MUST be sourced from the web using the available tools. If you don't know
    something, find it out using your tools.

- When you complete your task, your final response must be a comprehensive
    summary of your findings, presented in a clear, detailed, and
    easy-to-read format. Avoid using markdown tables for presenting data;
    use plain text formatting instead.



- You MUST actively use the query processing tools throughout your research:
    1. After each search, you MUST call `extract_relevant_details` to capture **all relevant details** 
       from both structured (infoboxes, lists, highlights) and unstructured (main text, later paragraphs) 
       content of the page. Do not stop at the first relevant snippet.
    2. You MUST regularly call `analyze_search_progress` to explicitly check if the evidence so far 
       covers all required information units in the question (e.g., full date with day, month, year).
    3. If any required unit is incomplete, missing, or uncertain, you MUST call query refinement/expansion 
       tools (`propose_query_refinement`, `local_refine_query`, `global_refine_query`, or `global_expand_query`) 
       to generate new candidate queries targeting the missing details.
    4. You MUST then use `select_query` to pick one query from the frontier and continue searching. 
       This iterative loop (extract → analyze → refine/expand → search) MUST continue until all gaps are resolved.

- Before concluding your research, you MUST verify completeness:
    1. You MUST call `analyze_search_progress` as the final checkpoint to confirm whether every 
       required element of the question is satisfied.
    2. If any gap remains, you MUST generate a refined/expanded query using the appropriate tool 
       (`propose_query_refinement`, `local_refine_query`, `global_refine_query`, or `global_expand_query`) 
       and continue searching until the gap is closed.
    3. You may only stop when:
       - All required information units are covered,
       - At least two independent sources (or one clearly authoritative source) confirm the information,
       - And your extracted notes show you have checked both structured and unstructured content.

</mandatory_instructions>

<capabilities>
Your capabilities include:
- Search and get information from the web using the search tools.
- Use the rich browser related toolset to investigate websites.
- Use the terminal tools to perform local operations. You can leverage
    powerful CLI tools like `grep` for searching within files, `curl` and
    `wget` for downloading content, and `jq` for parsing JSON data from APIs.
- Use the note-taking tools to record your findings.
</capabilities>

<web_search_workflow>
- Initial Search: You MUST start with a search engine like `search_google` to
    get a list of relevant URLs for your research, the URLs here will be used
    for `browser_visit_page`.
- Browser-Based Exploration: Use the rich browser related toolset to
    investigate websites.
    - **Navigation and Exploration**: Use `browser_visit_page` to open a URL.
        Navigate with `browser_click`, `browser_back`, and 
        `browser_forward`. Manage multiple pages with `browser_switch_tab`.
    - **Analysis**: Use `browser_get_som_screenshot` to understand the page 
        layout and identify interactive elements. Since this is a heavy 
        operation, only use it when visual analysis is necessary.
    - **Interaction**: Use `browser_type` to fill out forms and 
        `browser_enter` to submit or confirm search.

- In your response, you should mention the URLs you have visited and processed.
</web_search_workflow>

<query_processing_tools>
You have six tools to help you iteratively improve your search and ensure completeness:

**Query Management:**
1. **select_query**: Select a query from the current frontier to search. The frontier contains candidate queries that have been added by refine/expand tools. After selection, the query moves to the explored set. You should then call search_google with it (optionally adding search operators like site:, filetype:, quotes).

**Information Tracking:**
2. **extract_relevant_details**: After visiting a page, use this to confirm the specific information 
   you extracted that addresses the question. You provide the relevant details you identified, 
   and the tool will return them back to you.

3. **analyze_search_progress**: Use this to structure your evaluation of search completeness. 
   You write your analysis comparing what you've found against what the question requires, 
   and the tool returns your analysis back to you.

**Query Refinement and Expansion:**
4. **propose_query_refinement (LOCAL EXPAND)**: When you've identified a specific information gap 
   based on search results, use this to formulate a refined search query. Based on what you know 
   and what's missing, you propose a more targeted query, and the tool confirms it for your next search.

5. **local_refine_query (LOCAL REFINE)**: When your search results are insufficient but you want 
   to search for the same information with better phrasing, use this tool. It helps you rephrase 
   the query using different wording while maintaining the same search intent.

6. **global_refine_query (GLOBAL REFINE)**: When you identify issues with your query that you can 
   fix through your own understanding (without relying on search results), use this tool. It helps 
   improve query clarity, remove ambiguity, or fix formulation problems.

7. **global_expand_query (GLOBAL EXPAND)**: When you want to broaden your search scope using your 
   own knowledge, use this tool to add synonyms, related terms, or context to improve search coverage.

The query processing toolkit maintains a frontier of candidate queries and an explored set of searched queries. The frontier contains queries awaiting selection (added by refine/expand tools). The explored set contains queries already searched (moved from frontier after selection). You should track both sets during your research process.

Key considerations for thorough research:
- Ensure your findings completely answer what the question asks for, not just related information.
- If the question requires specific details, verify you have that level of precision.
- Choose the right refinement approach: LOCAL tools use search results as evidence, GLOBAL tools use your understanding.
- REFINE tools maintain the same search intent with better phrasing, EXPAND tools change scope or target gaps.
- All refine/expand tools add new queries to the frontier; use select_query to choose one for searching
- These tools help structure your iterative search process - use them to track your progress 
  and systematically fill information gaps.
</query_processing_tools>
"""
)


@api_keys_required(
    [
        (None, "GOOGLE_API_KEY"),
        (None, "SEARCH_ENGINE_ID"),
    ]
)
def deep_search_agent_factory(
    model: BaseModelBackend,
    working_directory: str,
):
    r"""Factory for creating a search agent, based on user-provided code
    structure.
    """

    environment = DeepSearchEnvironment(working_directory=working_directory)

    return DeepSearchAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Search Agent",
            content=SYSTEM_PROMPT(working_directory),
        ),
        model=model,
        environment=environment,
        prune_tool_calls_from_memory=True,
    )


@track_agent(name="SearchAgent")
class DeepSearchAgent(ChatAgent):
    r"""A :class:`ChatAgent` that conducts deep search on a given question."""

    def __init__(
        self,
        system_message: str,
        model: BaseModelBackend,
        environment: DeepSearchEnvironment,
        *args,
        **kwargs,
    ):
        self.environment = environment
        super().__init__(
            system_message=system_message,
            model=model,
            tools=environment.construct_action_space(),
            toolkits_to_register_agent=[environment.browser_toolkit],
            *args,
            **kwargs,
        )
        self.current_query_toolkit = None

    def update_note_taking_directory(self, new_directory: Path):
        """Update the working directory for note-taking toolkit."""
        self.environment.update_note_taking_directory(new_directory)

    async def areset(self):
        """Cleans up resources."""
        super().reset()
        if hasattr(self, "environment") and self.environment:
            await self.environment.cleanup()

    def reset(self):
        """Synchronous reset."""
        import asyncio
        import nest_asyncio

        nest_asyncio.apply()
        asyncio.run(self.areset())
        # self.remove_tools(
        #     [
        #         tool.get_function_name()
        #         for tool in self.current_query_toolkit.get_tools()
        #     ]
        # )
        # self.current_query_toolkit = None

    async def astep(
        self, input_query: str, response_format: Optional[Type[BaseModel]] = None
    ) -> ChatAgentResponse:
        self.environment.initialize_query(input_query)
        search_response = await super().astep(
            input_query,
            # f"Initial query: {input_query}\n\n{self.current_query_toolkit.get_frontier_str()}",
            response_format=response_format,
        )
        return search_response
