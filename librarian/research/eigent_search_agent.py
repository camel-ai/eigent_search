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
"""Search Agent implementation based on eigent.py without Exa tool."""

import asyncio
import datetime
import platform
import uuid
import os
import logging

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.toolkits import (
    SearchToolkit,
    TerminalToolkit,
)
from camel.toolkits.note_taking_toolkit import NoteTakingToolkit
from camel.utils import api_keys_required
from camel.toolkits.message_integration import ToolkitMessageIntegration
from camel.responses import ChatAgentResponse

from librarian.research.researcher import ResearchResponse
from librarian.research.browser_wrapper import BrowserToolkitWrapper

# Create timestamped temporary directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
WORKING_DIRECTORY = os.path.join(os.getcwd(), "tmp", f"eigent_{timestamp}")
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

logger = logging.getLogger(__name__)

import asyncio, threading
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()

def send_message_to_user(message: str):
    """Simple message handler for toolkit integration."""
    print(f"[Agent Message]: {message}")

def search_google_no_huggingface(query: str, **kwargs):
    r"""Use Google search engine to search information for the given query.

    Args:
        query (str): The query to be searched.
        search_type (str): The type of search to perform. Either "web" for
            web pages or "image" for image search. (default: "web")
        number_of_result_pages (int): The number of result pages to
            retrieve. Adjust this based on your task - use fewer results
            for focused searches and more for comprehensive searches.
            (default: :obj:`10`)
        start_page (int): The result page to start from. Use this for
            pagination - e.g., start_page=1 for results 1-10,
            start_page=11 for results 11-20, etc. This allows agents to
            check initial results and continue searching if needed.
            (default: :obj:`1`)

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary
        represents a search result.

            For web search, each dictionary contains:
            - 'result_id': A number in order.
            - 'title': The title of the website.
            - 'description': A brief description of the website.
            - 'long_description': More detail of the website.
            - 'url': The URL of the website.

            For image search, each dictionary contains:
            - 'result_id': A number in order.
            - 'title': The title of the image.
            - 'image_url': The URL of the image.
            - 'display_link': The website hosting the image.
            - 'context_url': The URL of the page containing the image.
            - 'width': Image width in pixels (if available).
            - 'height': Image height in pixels (if available).

            Example web result:
            {
                'result_id': 1,
                'title': 'OpenAI',
                'description': 'An organization focused on ensuring that
                artificial general intelligence benefits all of humanity.',
                'long_description': 'OpenAI is a non-profit artificial
                intelligence research company. Our goal is to advance
                digital intelligence in the way that is most likely to
                benefit humanity as a whole',
                'url': 'https://www.openai.com'
            }

            Example image result:
            {
                'result_id': 1,
                'title': 'Beautiful Sunset',
                'image_url': 'https://example.com/image.jpg',
                'display_link': 'example.com',
                'context_url': 'https://example.com/page.html',
                'width': 800,
                'height': 600
            }
    """

    from camel.toolkits import SearchToolkit
    
    search_toolkit = SearchToolkit()
    query_with_filter = f"{query} -site:huggingface.co -site:hf.co -site:oxen.ai"
    
    results = search_toolkit.search_google(query_with_filter, **kwargs)
    
    if isinstance(results, str):
        lines = results.split('\n')
        filtered_lines = []
        for line in lines:
            if ('huggingface.co' not in line.lower() and 
                'hf.co' not in line.lower() and 
                'oxen.ai' not in line.lower()):
                filtered_lines.append(line)
        return '\n'.join(filtered_lines)
    
    return results

@api_keys_required(
    [
        (None, 'GOOGLE_API_KEY'),
        (None, 'SEARCH_ENGINE_ID')
    ]
)
def search_agent_factory(
    model: BaseModelBackend,
    task_id: str,
) -> tuple[ChatAgent, BrowserToolkitWrapper]:
    r"""Factory for creating a search agent, based on user-provided code
    structure.
    """
    # Initialize message integration
    # message_integration = ToolkitMessageIntegration(
        # message_handler=send_message_to_user
    # )

    # Generate a unique identifier for this agent instance
    agent_id = str(uuid.uuid4())[:8]

    custom_tools = [
        "browser_open",
        "browser_close",
        "browser_back",
        "browser_forward",
        "browser_click",
        # "browser_type",
        # "browser_enter",
        "browser_visit_page",
        "browser_get_tab_info",
        "browser_close_tab",
        "browser_switch_tab",
        "browser_get_som_screenshot",
    ]
    web_toolkit_custom = BrowserToolkitWrapper(
        mode="python",
        headless=True,
        enabled_tools=custom_tools,
        browser_log_to_file=True,
        stealth=True,
        session_id=agent_id,
        viewport_limit=False,
        cache_dir=WORKING_DIRECTORY,
        default_start_url="https://search.brave.com/",
        domain_blacklist=['huggingface.co', 'hf.co', 'oxen.ai'],  # Add more domains here as needed
    )

    # Initialize toolkits
    terminal_toolkit = TerminalToolkit(safe_mode=True, clone_current_env=False)
    note_toolkit = NoteTakingToolkit(working_directory=WORKING_DIRECTORY)
    # search_toolkit = SearchToolkit()
    # terminal_toolkit_basic = TerminalToolkit()

    # Add messaging to toolkits
    # web_toolkit_custom = message_integration.register_toolkits(
    #     web_toolkit_custom
    # )
    # terminal_toolkit = message_integration.register_toolkits(terminal_toolkit)
    # note_toolkit = message_integration.register_toolkits(note_toolkit)
    # enhanced_shell_exec = message_integration.register_functions(
    #     [terminal_toolkit_basic.shell_exec]
    # )

    tools = [
        *web_toolkit_custom.get_tools(),
        # *enhanced_shell_exec,
        *note_toolkit.get_tools(),
        search_google_no_huggingface,  # Use wrapper that blocks Hugging Face results
        *terminal_toolkit.get_tools(),
    ]

    system_message = f"""
<role>
You are a Senior Research Analyst, a key member of a multi-agent team. Your 
primary responsibility is to conduct expert-level web research to gather, 
analyze, and document information required to solve the user's task. You 
operate with precision, efficiency, and a commitment to data quality.
</role>

<operating_environment>
- **System**: {platform.system()} ({platform.machine()})
- **Working Directory**: `{WORKING_DIRECTORY}`. All local file operations must
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
    1. Returned by a search tool (like `search_google`, `search_bing`,
        or `search_exa`).
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
- You MUST NOT visit the same URL more than once.

- In your response, you should mention the URLs you have visited and processed.

</web_search_workflow>
"""

    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Search Agent",
            content=system_message,
        ),
        model=model,
        toolkits_to_register_agent=[web_toolkit_custom],
        tools=tools,
        prune_tool_calls_from_memory=True,
        max_iteration=30,
        message_window_size=100000,
    ), web_toolkit_custom


class EigentSearchAgent:
    """Wrapper class for the Eigent Search Agent with async support."""
    
    def __init__(self, model: BaseModelBackend):
        """Initialize the Eigent Search Agent.
        
        Args:
            model: The model backend to use for the agent.
        """
        self.model = model
        self.task_id = str(uuid.uuid4())[:8]
        self.agent = None
        self.web_toolkit = None
        self.reset()
    
    def _run_async_in_sync(self, coro):
        """Helper method to run async coroutine from sync context.
        
        This handles the case where there's already a running event loop
        by using run_coroutine_threadsafe.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            The result of the coroutine
        """
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result()
                

    
    def reset(self):
        """Reset the agent by creating a new instance.
        
        This method:
        1. Resets the agent state
        2. Resets the browser wrapper (closes browser and clears history)
        3. Creates a new agent and browser toolkit instance
        """
        if self.agent is not None:
            self.agent.reset()
            
        if self.web_toolkit is not None:
            try:
                self._run_async_in_sync(self.web_toolkit.reset())
            except Exception as e:
                logger.warning(f"Could not reset browser wrapper: {e}")
        
        # Create new instances
        self.agent, self.web_toolkit = search_agent_factory(self.model, self.task_id)
        # if self.web_toolkit is None:
            # self.web_toolkit = web_toolkit_new
    
    def step(self, task_prompt: str) -> ResearchResponse:
        """Synchronous step method for compatibility.
        
        Args:
            task_prompt: The task prompt to send to the agent.
            
        Returns:
            The agent's response.
        """
        return self._run_async_in_sync(self.agent.astep(task_prompt, response_format=ResearchResponse))
    