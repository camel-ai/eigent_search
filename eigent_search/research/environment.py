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

from camel.logger import get_logger
from camel.toolkits import (
    FunctionTool,
    HybridBrowserToolkit,
    NoteTakingToolkit,
    SearchToolkit,
    TerminalToolkit,
    ToolkitMessageIntegration,
)


logger = get_logger(__name__)


class DeepSearchEnvironment:
    """Toolkit for conducting deep search on a given question."""

    def __init__(self, working_directory: str):
        self.environment_id = str(uuid.uuid4())[:8]
        self.working_directory = working_directory

        # Construct toolkits
        self.search_toolkit = self.construct_search_toolkit()
        self.browser_toolkit = self.construct_browser_toolkit()
        self.terminal_toolkit = self.construct_terminal_toolkit()
        self.note_taking_toolkit = self.construct_note_taking_toolkit()
        self.message_integration = self.construct_message_integration()

        # Add messaging to toolkits
        self.search_toolkit = self.message_integration.register_toolkits(
            self.search_toolkit
        )
        self.browser_toolkit = self.message_integration.register_toolkits(
            self.browser_toolkit
        )
        self.terminal_toolkit = self.message_integration.register_toolkits(
            self.terminal_toolkit
        )
        self.note_taking_toolkit = self.message_integration.register_toolkits(
            self.note_taking_toolkit
        )

    def construct_action_space(self):
        """Construct a toolkit for actions related to the deep search environment."""
        tools = [
            *self.browser_toolkit.get_tools(),
            *self.note_taking_toolkit.get_tools(),
            *self.search_toolkit.get_tools(),
            *self.terminal_toolkit.get_tools(),
        ]
        return tools

    def construct_search_toolkit(
        self, exclude_domains: list[str] = ["huggingface.co", "hf.co", "oxen.ai"]
    ):
        """Construct a search toolkit for actions related to searching the web."""

        search_toolkit = SearchToolkit(exclude_domains=exclude_domains)
        # Only search_google is needed, so we override the get_tools method
        search_toolkit.get_tools = lambda: [FunctionTool(search_toolkit.search_google)]
        return search_toolkit

    def construct_browser_toolkit(self):
        """Construct a browser toolkit for actions related to browsing the web."""

        custom_tools = [
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
        web_toolkit_custom = HybridBrowserToolkit(
            mode="python",
            headless=True,
            enabled_tools=custom_tools,
            browser_log_to_file=True,
            stealth=True,
            session_id=self.environment_id,
            viewport_limit=False,
            log_dir=os.path.join(self.working_directory, "browser_logs"),
            cache_dir=self.working_directory,
            default_start_url="https://search.brave.com/",
        )
        return web_toolkit_custom

    def construct_terminal_toolkit(self):
        """Construct a terminal toolkit for actions related to terminal operations."""
        terminal_toolkit = TerminalToolkit(
            safe_mode=True,
            clone_current_env=False,
            log_dir=os.path.join(self.working_directory, "terminal_logs"),
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
                FunctionTool(terminal_toolkit.shell_wait),
                FunctionTool(terminal_toolkit.shell_write_to_process),
                FunctionTool(terminal_toolkit.shell_kill_process),
                # FunctionTool(terminal_toolkit.ask_user_for_help),
            ]

        terminal_toolkit.get_tools = custom_get_tools
        return terminal_toolkit

    def construct_note_taking_toolkit(self):
        """Construct a note toolkit for actions related to note-taking."""
        return NoteTakingToolkit(
            working_directory=os.path.join(self.working_directory, "note_taking_logs")
        )

    def construct_message_integration(self):
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
