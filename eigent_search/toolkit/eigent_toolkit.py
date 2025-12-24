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

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from camel.logger import get_logger
from camel.toolkits import (
    FunctionTool,
    HybridBrowserToolkit,
    NoteTakingToolkit,
    SearchToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
)

from eigent_search.toolkit import CleanupToolkit

logger = get_logger(__name__)


class EigentSearchToolkit(CleanupToolkit):
    """Default Eigent search toolkit implementation."""

    def __init__(
        self,
        working_directory: Path,
        exclude_search_domains: list[str] = ["huggingface.co", "hf.co", "oxen.ai"],
        use_customized_search_google: bool = False,
    ):
        super().__init__()
        # Basic session information
        self.working_directory = working_directory
        self.session_id = str(uuid.uuid4())[:8]
        self.exclude_search_domains = exclude_search_domains
        self.use_customized_search_google = use_customized_search_google

        # Construct toolkits
        self.search_toolkit = self._construct_search_toolkit()
        self.browser_toolkit = self._construct_browser_toolkit()
        self.terminal_toolkit = self._construct_terminal_toolkit()
        self.note_taking_toolkit = self._construct_note_taking_toolkit()
        self.message_integration = self._construct_message_integration()
        # Initialize query processing toolkit (will be created with specific query when needed)
        self.query_processing_toolkit = None

        # Message registration function for toolkits
        self.register = lambda toolkit: self.message_integration.register_toolkits(
            toolkit
        )

        # Add messaging to toolkits
        self.search_toolkit = self.register(self.search_toolkit)
        self.browser_toolkit = self.register(self.browser_toolkit)
        self.terminal_toolkit = self.register(self.terminal_toolkit)
        self.note_taking_toolkit = self.register(self.note_taking_toolkit)

    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if hasattr(self.browser_toolkit, "browser_close"):
                await self.browser_toolkit.browser_close()
                logger.info("Browser closed successfully during cleanup.")
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")

    def get_tools(self) -> list[FunctionTool]:
        """Get the tools for the eigent search toolkit."""
        return [
            *self.search_toolkit.get_tools(),
            *self.browser_toolkit.get_tools(),
            *self.terminal_toolkit.get_tools(),
            *self.note_taking_toolkit.get_tools(),
        ]

    def _construct_search_toolkit(self):
        """Construct a search toolkit for actions related to searching the web."""

        search_toolkit = SearchToolkit(exclude_domains=self.exclude_search_domains)
        use_customized = self.use_customized_search_google
        # Capture exclude_domains via closure
        exclude_domains = search_toolkit.exclude_domains

        def search_google(
            query: str,
            search_type: str = "web",
            number_of_result_pages: int = 10,
            start_page: int = 1,
        ) -> List[Dict[str, Any]]:
            r"""Use Google search engine to search information for the given query.

            Args:
                query (str): The query to be searched.
                search_type (str): The type of search to perform. Must be either
                    "web" for web pages or "image" for image search. Any other
                    value will raise a ValueError. (default: "web")
                number_of_result_pages (int): The number of result pages to
                    retrieve. Must be a positive integer between 1 and 10.
                    Google Custom Search API limits results to 10 per request.
                    If a value greater than 10 is provided, it will be capped
                    at 10 with a warning. Adjust this based on your task - use
                    fewer results for focused searches and more for comprehensive
                    searches. (default: :obj:`10`)
                start_page (int): The result page to start from. Must be a
                    positive integer (>= 1). Use this for pagination - e.g.,
                    start_page=1 for results 1-10, start_page=11 for results
                    11-20, etc. This allows agents to check initial results
                    and continue searching if needed. (default: :obj:`1`)

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
            from urllib.parse import quote

            import requests

            # Override parameters with fixed values if customized search is enabled
            if use_customized:
                search_type = "web"
                number_of_result_pages = 10
                start_page = 1

            # Validate input parameters
            if not isinstance(start_page, int) or start_page < 1:
                raise ValueError("start_page must be a positive integer")

            if (
                not isinstance(number_of_result_pages, int)
                or number_of_result_pages < 1
            ):
                raise ValueError("number_of_result_pages must be a positive integer")

            # Google Custom Search API has a limit of 10 results per request
            if number_of_result_pages > 10:
                logger.warning(
                    f"Google API limits results to 10 per request. "
                    f"Requested {number_of_result_pages}, using 10 instead."
                )
                number_of_result_pages = 10

            if search_type not in ["web", "image"]:
                raise ValueError("search_type must be either 'web' or 'image'")

            # https://developers.google.com/custom-search/v1/overview
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            # https://cse.google.com/cse/all
            SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

            # Using the specified start page
            start_page_idx = start_page
            # Different language may get different result
            search_language = "en"

            modified_query = query
            if exclude_domains:
                # Use Google's -site: operator to exclude domains
                exclusion_terms = " ".join(
                    [f"-site:{domain}" for domain in exclude_domains]
                )
                modified_query = f"{exclusion_terms} {query}"
                logger.debug(f"Excluded domains, modified query: {modified_query}")

            encoded_query = quote(modified_query)
            # encoded_query = modified_query
            # Constructing the URL
            # Doc: https://developers.google.com/custom-search/v1/using_rest
            base_url = (
                f"https://www.googleapis.com/customsearch/v1?"
                f"key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={encoded_query}&start="
                f"{start_page_idx}&lr={search_language}&num={number_of_result_pages}"
            )

            # Add searchType parameter for image search
            if search_type == "image":
                url = base_url + "&searchType=image"
            else:
                url = base_url

            responses = []
            # Fetch the results given the URL
            try:
                # Make the get
                result = requests.get(url)
                data = result.json()

                # Get the result items
                if "items" in data:
                    search_items = data.get("items")

                    # Iterate over results found
                    for i, search_item in enumerate(search_items, start=1):
                        if search_type == "image":
                            # Process image search results
                            title = search_item.get("title")
                            image_url = search_item.get("link")
                            display_link = search_item.get("displayLink")

                            # Get context URL (page containing the image)
                            image_info = search_item.get("image", {})
                            context_url = image_info.get("contextLink", "")

                            # Get image dimensions if available
                            width = image_info.get("width")
                            height = image_info.get("height")

                            response = {
                                "result_id": i,
                                "title": title,
                                "image_url": image_url,
                                "display_link": display_link,
                                "context_url": context_url,
                            }

                            if width:
                                response["width"] = int(width)
                            if height:
                                response["height"] = int(height)

                            responses.append(response)
                        else:
                            if "pagemap" not in search_item:
                                continue
                            if "metatags" not in search_item["pagemap"]:
                                continue
                            if (
                                "og:description"
                                in search_item["pagemap"]["metatags"][0]
                            ):
                                long_description = search_item["pagemap"]["metatags"][
                                    0
                                ]["og:description"]
                            else:
                                long_description = "N/A"
                            title = search_item.get("title")
                            snippet = search_item.get("snippet")

                            link = search_item.get("link")
                            response = {
                                "result_id": i,
                                "title": title,
                                "description": snippet,
                                "long_description": long_description,
                                "url": link,
                            }
                            responses.append(response)
                else:
                    if "error" in data:
                        error_info = data.get("error", {})
                        logger.error(
                            f"Google search failed - API response: {error_info}"
                        )
                        responses.append(
                            {
                                "error": f"Google search failed - "
                                f"API response: {error_info}"
                            }
                        )
                    elif "searchInformation" in data:
                        search_info = data.get("searchInformation", {})
                        total_results = search_info.get("totalResults", "0")
                        if total_results == "0":
                            logger.info(f"No results found for query: {query}")
                            # Return empty list to indicate no results (not an error)
                            responses = []
                        else:
                            logger.warning(
                                f"Google search returned no items but claims {total_results} results"
                            )
                            responses = []
                    else:
                        logger.error(f"Unexpected Google API response format: {data}")
                        responses.append(
                            {"error": "Unexpected response format from Google API"}
                        )

            except Exception as e:
                responses.append({"error": f"google search failed: {e!s}"})
            return responses

        search_toolkit.search_google = search_google
        # Only search_google is needed, so we override the get_tools method
        search_toolkit.get_tools = lambda: [FunctionTool(search_toolkit.search_google)]
        return search_toolkit

    def _construct_browser_toolkit(self):
        """Construct a browser toolkit for actions related to browsing the web."""

        enabled_tools = [
            "browser_open",
            "browser_close",
            "browser_back",
            "browser_forward",
            "browser_click",
            "browser_type",
            "browser_enter",
            "browser_switch_tab",
            "browser_visit_page",
            "browser_get_som_screenshot",
        ]
        browser_toolkit = HybridBrowserToolkit(
            mode="typescript",
            headless=True,
            enabled_tools=enabled_tools,
            browser_log_to_file=True,
            stealth=True,
            session_id=self.session_id,
            viewport_limit=False,
            log_dir=os.path.join(self.working_directory.as_posix(), "browser_logs"),
            cache_dir=os.path.join(self.working_directory.as_posix(), "browser_logs"),
            default_start_url="https://search.brave.com/",
        )
        return browser_toolkit

    def _construct_terminal_toolkit(self):
        """Construct a terminal toolkit for actions related to terminal operations."""
        terminal_toolkit = TerminalToolkit(
            working_directory=self.working_directory / "terminal_workspace",
            safe_mode=True,
            clone_current_env=False,
            session_logs_dir=self.working_directory / "terminal_logs",
        )

        # Override get_tools method to only include specific tools
        def custom_get_tools() -> list[FunctionTool]:
            r"""Returns a list of FunctionTool objects representing the functions
            in the toolkit.

            Returns:
                List[FunctionTool]: A list of FunctionTool objects representing the
                    functions in the toolkit.
            """
            return [
                FunctionTool(terminal_toolkit.shell_exec),
                FunctionTool(terminal_toolkit.shell_view),
                # FunctionTool(terminal_toolkit.shell_wait),
                FunctionTool(terminal_toolkit.shell_write_to_process),
                FunctionTool(terminal_toolkit.shell_kill_process),
                # FunctionTool(terminal_toolkit.ask_user_for_help),
            ]

        terminal_toolkit.get_tools = custom_get_tools
        return terminal_toolkit

    def _construct_note_taking_toolkit(self):
        """Construct a note toolkit for actions related to note-taking."""
        return NoteTakingToolkit(
            working_directory=os.path.join(
                self.working_directory.as_posix(), "note_taking_logs"
            )
        )

    def _construct_message_integration(self):
        """Construct a message integration toolkit to allow agents to send status updates to users"""

        # TODO: Doc string needs to be rewritten to fit search context
        def send_message_to_user(
            message_title: str,
            message_description: str,
            message_attachment: str = "",
        ) -> str:
            r"""Use this tool to send a tidy message to the user, including a
            short title, a one-sentence description, and an optional attachment.

            This one-way tool keeps the user informed about your progress,
            decisions, or actions. It does not require a response.
            You should use it to:
            - Announce what you are about to do.
            For example:
            message_title="Starting Task"
            message_description="Searching for papers on GUI Agents."
            - Report the result of an action.
            For example:
            message_title="Search Complete"
            message_description="Found 15 relevant papers."
            - Report a created file.
            For example:
            message_title="File Ready"
            message_description="The report is ready for your review."
            message_attachment="report.pdf"
            - State a decision.
            For example:
            message_title="Next Step"
            message_description="Analyzing the top 10 papers."
            - Give a status update during a long-running task.

            Args:
                message_title (str): The title of the message.
                message_description (str): The short description.
                message_attachment (str): The attachment of the message,
                    which can be a file path or a URL.

            Returns:
                str: Confirmation that the message was successfully sent.
            """
            print(f"\nAgent Message:\n{message_title} \n{message_description}\n")
            if message_attachment:
                print(message_attachment)
            logger.info(
                f"\nAgent Message:\n{message_title} "
                f"{message_description} {message_attachment}"
            )
            return (
                f"Message successfully sent to user: '{message_title} "
                f"{message_description} {message_attachment}'"
            )

        return ToolkitMessageIntegration(message_handler=send_message_to_user)
