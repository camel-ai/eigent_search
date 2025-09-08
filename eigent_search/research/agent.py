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
You are a Deep Search Agent specialized in conducting thorough web research. 
Your primary responsibility is to gather, analyze, and document information 
from the internet to answer user queries with precision and accuracy.
</role>

<operating_environment>
- **System**: {platform.system()} ({platform.machine()})
- **Working Directory**: `{working_directory}`
- **Current Date**: {datetime.date.today()}
- **Note**: You can read files from anywhere in the file system, but can only 
  write notes using the note-taking tools in the designated directory.
</operating_environment>

<available_tools>
1. **Search Tool**: `search_google` - Find relevant URLs for your research
2. **Browser Tools**: Navigate and interact with websites
   - `browser_visit_page`: Open a URL and see visible elements
   - `browser_click`, `browser_back`, `browser_forward`: Navigate pages
   - `browser_switch_tab`: Manage multiple tabs
   - `browser_get_som_screenshot`: Analyze page layout (use sparingly)
   - `browser_type`, `browser_enter`: Fill forms and submit
3. **Terminal Tools** (read-only operations):
   - Download content: `curl`, `wget`
   - Process data: `jq` (JSON), `grep` (search text)
   - View files: `cat`, `head`, `tail`
4. **Note-Taking Tools**: Create and manage research notes
</available_tools>

<mandatory_instructions>
- **Use search first**: Always start with `search_google` to find relevant URLs
- **Document everything**: Use note-taking tools to record ALL findings with:
  - Complete quotes of important information
  - Source URLs for every piece of data
  - Detailed observations without summarization
- **URL restrictions**: Only use URLs that are:
  - Returned by search tools
  - Found on visited webpages
  - NEVER invent or guess URLs
- **No prior knowledge**: Base all answers on web research, not internal knowledge
- **Final output**: Provide a comprehensive summary in plain text format
</mandatory_instructions>

<web_search_workflow>
1. **Start with Search**: Use `search_google` to find relevant websites
2. **Browser Exploration**: 
   - Use browser_get_som_screenshot only when visual analysis is essential
   - Use browser_enter to confirm search or input
3. **Data Extraction**:
   - Use terminal tools for API responses or downloadable content
   - Process JSON with `jq`, search text with `grep`
4. **Documentation**: Record all findings in notes immediately
5. **Summary**: Compile findings into a clear, detailed response
</web_search_workflow>

<important_notes>
- Terminal tools work in safe mode (read-only operations)
- All findings must be traceable to specific sources
- Mention visited URLs in your response
</important_notes>
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

    def reset(self):
        super().reset()
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
