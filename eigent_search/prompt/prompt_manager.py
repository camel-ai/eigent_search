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
    from eigent_search.config import AblationType, SearchAgentType


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
        self,
        working_directory: str | Path,
        agent_type: SearchAgentType,
        ablation_type: AblationType | None = None,
    ) -> str:
        """Get the preset system prompt for the given agent type and ablation.

        Args:
            working_directory: Working directory for the agent
            agent_type: The type of search agent
            ablation_type: Optional ablation configuration for experiments

        Returns:
            Rendered system prompt string
        """
        # Import here to avoid circular imports
        from eigent_search.config import AblationType

        if agent_type.value == "search_only":
            return self.get_system_prompt("search_only", working_directory)
        elif agent_type.value == "eigent_search":
            return self.get_system_prompt("eigent_search", working_directory)
        elif agent_type.value == "eigent_search_q+":
            # Handle ablation-specific prompts
            if ablation_type is None or ablation_type == AblationType.NONE:
                return self.get_system_prompt("eigent_search_q+", working_directory)
            elif ablation_type == AblationType.FIXED_10_RESULTS:
                return self.get_system_prompt(
                    "eigent_search_q+_fixed_10_results", working_directory
                )
            elif ablation_type == AblationType.FIXED_10_RESULTS_EIGENT_PROMPT:
                return self.get_system_prompt(
                    "eigent_search_q+_fixed_10_results_eigent_prompt", working_directory
                )
            elif ablation_type == AblationType.NO_QUERY_TOOLS:
                return self.get_system_prompt(
                    "eigent_search_q+_no_query_tools", working_directory
                )
            elif ablation_type == AblationType.QUERY_TOOLS_ONLY:
                return self.get_system_prompt(
                    "eigent_search_q+_query_tools_only", working_directory
                )
            else:
                return self.get_system_prompt("eigent_search_q+", working_directory)


# Global instance for easy access
prompt_manager = PromptManager()
