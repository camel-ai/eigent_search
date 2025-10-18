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
"""System prompt template manager using Jinja2."""

from __future__ import annotations
import datetime
from pathlib import Path
import platform
from typing import Any
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

# Import SearchAgentType from config to avoid circular imports
if TYPE_CHECKING:
    from eigent_search.config import SearchAgentType


class PromptManager:
    """Manages system prompt templates using Jinja2."""

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize the prompt manager.

        Args:
            template_dir: Directory containing Jinja2 templates.
                         Defaults to templates directory in the same package.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent

        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_system_prompt(
        self,
        template_name: str,
        working_directory: str | Path,
        **kwargs: Any,
    ) -> str:
        """Generate a system prompt from a specific template.

        Args:
            template_name: Name of the template file (without .j2 extension)
            working_directory: Working directory for the agent
            **kwargs: Additional template variables

        Returns:
            Rendered system prompt string
        """
        template = self.env.get_template(f"{template_name}.md")

        # Default context
        context = {
            "working_directory": str(working_directory),
            "system_info": platform.system(),
            "machine_info": platform.machine(),
            "current_date": datetime.date.today(),
            **kwargs,
        }

        return template.render(context)

    def get_preset_system_prompt(
        self, working_directory: str | Path, agent_type: SearchAgentType
    ) -> str:
        """Get the preset system prompt for the given agent type."""
        if agent_type.value == "search_only":
            return self.get_system_prompt("search_only", working_directory)
        elif agent_type.value == "eigent_search":
            return self.get_system_prompt("eigent_search", working_directory)
        elif agent_type.value == "eigent_search_q+":
            return self.get_system_prompt("eigent_search_q+", working_directory)


# Global instance for easy access
prompt_manager = PromptManager()
