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
    1. Returned by a search tool (like `select_query_and_search`).
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
    1. Use `select_query_and_search` to select and search queries from the frontier
    2. After each search, use `extract_relevant_details` to document findings
    3. Regularly call `analyze_search_progress` to verify completeness
    4. Use query refinement/expansion tools when gaps are identified

- Before concluding your research, you MUST verify completeness:
    1. Call `analyze_search_progress` as the final checkpoint
    2. If any gap remains, generate refined/expanded queries and continue searching
    3. Only stop when all required information is covered with sufficient evidence

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
- Initial Search: You MUST start with `select_query_and_search` to
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
You have six tools to help you iteratively improve your search:

**Search and Selection:**
1. **select_query_and_search**: Select a query from the frontier and perform web search

**Information Tracking:**
2. **extract_relevant_details**: Document specific information extracted from pages
3. **analyze_search_progress**: Evaluate whether findings answer the question completely

**Query Refinement and Expansion:**
4. **local_expand_query**: Generate multiple queries targeting identified information gaps
5. **local_refine_query**: Generate multiple rephrased queries with same intent but better wording
6. **global_refine_query**: Generate multiple improved queries using your understanding
7. **global_expand_query**: Generate multiple expanded queries with additional terms

All query tools help structure your iterative search process. The toolkit maintains:
- **Frontier**: Candidate queries awaiting search (added by refine/expand tools)
- **Explored**: Queries already searched (moved from frontier after selection)

**Important Notes:**
- `select_query_and_search` now takes only ONE parameter: the query from the frontier
- If you want to use search operators (AND, OR, NOT, quotes, site:, filetype:, etc.), 
  create a refined or expanded query using the query and expansion refinement tools first, then select it
- Each call to `select_query_and_search` moves the selected query from frontier to explored
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

    async def astep(
            self, input_query: str, response_format: Optional[Type[BaseModel]] = None
    ) -> ChatAgentResponse:
        self.environment.initialize_query(input_query)
        search_response = await super().astep(
            input_query,
            response_format=response_format,
        )
        return search_response