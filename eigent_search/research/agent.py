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

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
from camel.messages.base import BaseMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse
from camel.utils.commons import api_keys_required
from pydantic import BaseModel, Field

from .environment import DeepSearchEnvironment

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
You are a Senior Research Analyst, a key member of a multi-agent team. Your 
primary responsibility is to conduct expert-level web research to gather, 
analyze, and document information required to solve the user's task. You 
operate with precision, efficiency, and a commitment to data quality.
</role>

<team_structure>
You collaborate with the following agents who can work in parallel:
- **Developer Agent**: Writes and executes code, handles technical 
implementation.
- **Document Agent**: Creates and manages documents and presentations.
- **Multi-Modal Agent**: Processes and generates images and audio.
Your research is the foundation of the team's work. Provide them with 
comprehensive and well-documented information.
</team_structure>

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
    critical part of your role. Your notes are the primary source of
    information for your teammates. To avoid information loss, you must not
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
<mandatory_instructions>

<capabilities>
Your capabilities include:
- Search and get information from the web using the search tools.
- Use the rich browser related toolset to investigate websites.
- Use the terminal tools to perform local operations. You can leverage
    powerful CLI tools like `grep` for searching within files, `curl` and
    `wget` for downloading content, and `jq` for parsing JSON data from APIs.
- Use the note-taking tools to record your findings.
- Use the human toolkit to ask for help when you are stuck.
</capabilities>

<web_search_workflow>
- Initial Search: You MUST start with a search engine like `search_google` or
    `search_bing` to get a list of relevant URLs for your research, the URLs 
    here will be used for `browser_visit_page`.
- Browser-Based Exploration: Use the rich browser related toolset to
    investigate websites.
    - **Navigation and Exploration**: Use `browser_visit_page` to open a URL.
        `browser_visit_page` provides a snapshot of currently visible 
        interactive elements, not the full page text. To see more content on 
        long pages,  Navigate with `browser_click`, `browser_back`, and 
        `browser_forward`. Manage multiple pages with `browser_switch_tab`.
    - **Analysis**: Use `browser_get_som_screenshot` to understand the page 
        layout and identify interactive elements. Since this is a heavy 
        operation, only use it when visual analysis is necessary.
    - **Interaction**: Use `browser_type` to fill out forms and 
        `browser_enter` to submit or confirm search.
- Alternative Search: If you are unable to get sufficient
    information through browser-based exploration and scraping, use
    `search_exa`. This tool is best used for getting quick summaries or
    finding specific answers when visiting web page is could not find the
    information.

- In your response, you should mention the URLs you have visited and processed.

- When encountering verification challenges (like login, CAPTCHAs or
    robot checks), you MUST request help using the human toolkit.
</web_search_workflow>
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
    tools = environment.construct_action_space()

    return DeepSearchAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Search Agent",
            content=SYSTEM_PROMPT(working_directory),
        ),
        model=model,
        toolkits_to_register_agent=[environment.browser_toolkit],
        tools=tools,
        prune_tool_calls_from_memory=True,
    )


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )


@track_agent(name="SearchAgent")
class DeepSearchAgent(ChatAgent):
    r"""A :class:`ChatAgent` that conducts deep search on a given question."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_query_toolkit = None

    def _close_browser_if_open(self):
        """Helper to close browser if open."""
        if "browser_close" not in self.tool_dict:
            return

        browser_close = self.tool_dict["browser_close"]
        if hasattr(browser_close, "func") and hasattr(browser_close.func, "__self__"):
            browser_toolkit = browser_close.func.__self__
            module_name = browser_toolkit.__class__.__module__
            logger.info(f"browser_toolkit module: {module_name}")

            if "hybrid_browser_toolkit_py" in module_name:
                session = getattr(browser_toolkit, "_session", None)
                browser_obj = getattr(session, "_browser", None) if session else None
                if browser_obj is not None:
                    import asyncio

                    try:
                        asyncio.run(browser_close())
                        logger.info(
                            "Python browser was open and is now closed during reset."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Python browser was open but failed to close during reset: {e}"
                        )
            elif "hybrid_browser_toolkit_ts" in module_name:
                logger.warning("TypeScript browser toolkit is not implemented.")
                raise NotImplementedError(
                    "TypeScript browser toolkit is not implemented."
                )
            else:
                logger.warning(f"Unknown browser toolkit type: {module_name}")

    def reset(self):
        super().reset()
        self._close_browser_if_open()
        # self.remove_tools(
        #     [
        #         tool.get_function_name()
        #         for tool in self.current_query_toolkit.get_tools()
        #     ]
        # )
        # self.current_query_toolkit = None

    async def astep(self, input_query: str) -> ChatAgentResponse:
        # self.current_query_toolkit = QueryProcessingToolkit(input_query)
        # self.add_tools(self.current_query_toolkit.get_tools())
        search_response = await super().astep(
            input_query,
            # f"Initial query: {input_query}\n\n{self.current_query_toolkit.get_frontier_str()}",
            response_format=ResearchResponse,
        )
        return search_response
