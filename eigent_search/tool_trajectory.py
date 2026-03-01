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
from typing import Any

from camel.logger import get_logger
from camel.responses import ChatAgentResponse
from camel.types.agents.tool_calling_record import ToolCallingRecord
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class ToolCallInfo(BaseModel):
    """Information about a tool call.

    We discard the images field from `ToolCallingRecord` because it is not needed here.

    For reasoning models (e.g., DeepSeek R1, Minimax M2.5), reasoning_content captures
    the model's thinking process before making the tool call.
    """

    tool_call_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any
    reasoning_content: str | None = None  # For reasoning models

    @classmethod
    def extract_from_tool_call_record(
        cls, tool_call_index: int, tool_call_record: ToolCallingRecord, reasoning_content: str | None = None
    ) -> ToolCallInfo:
        return cls(
            tool_call_index=tool_call_index,
            tool_call_id=tool_call_record.tool_call_id,
            tool_name=tool_call_record.tool_name,
            arguments=tool_call_record.args,
            result=tool_call_record.result,
            reasoning_content=reasoning_content,
        )


class ToolTrajectory(BaseModel):
    tool_counts: dict[str, int]
    trajectory_length: int
    trajectory: list[ToolCallInfo]

    # Pattern that indicates Google API daily limit reached
    GOOGLE_API_LIMIT_PATTERN: str = "google search failed - api response"

    def has_google_api_limit_error(self) -> bool:
        """Check if Google API daily limit error occurred in tool results."""
        for tool_call in self.trajectory:
            result_str = str(tool_call.result).lower()
            if self.GOOGLE_API_LIMIT_PATTERN in result_str:
                return True
        return False

    @classmethod
    def extract_from_response(cls, response: ChatAgentResponse) -> ToolTrajectory:
        trajectory = []
        tool_counts = {}

        # Build a mapping of tool_call_id -> reasoning_content from messages
        # For interleaved thinking models (Minimax), each tool call has its own reasoning
        # For single-block reasoning models (DeepSeek R1), use the first message's reasoning
        tool_call_reasoning_map = {}

        if response.msgs:
            # Try to find reasoning for each message that has a tool_call_id
            from camel.messages import FunctionCallingMessage

            for msg in response.msgs:
                if isinstance(msg, FunctionCallingMessage) and hasattr(msg, 'tool_call_id'):
                    # Minimax-style: each FunctionCallingMessage has its own reasoning
                    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                        tool_call_reasoning_map[msg.tool_call_id] = msg.reasoning_content

            # Fallback for DeepSeek R1 style: use first message's reasoning for all
            if not tool_call_reasoning_map and len(response.msgs) > 0:
                first_msg = response.msgs[0]
                if hasattr(first_msg, 'reasoning_content') and first_msg.reasoning_content:
                    # Store it with None key to indicate it's the default
                    tool_call_reasoning_map[None] = first_msg.reasoning_content

        # CRITICAL: For Minimax M2.5, reasoning is stored in a global registry (not in response.msgs)
        # If we didn't find reasoning in response.msgs, try the Minimax registry
        if not tool_call_reasoning_map:
            try:
                from eigent_search.minimax_m25_patch import get_minimax_content
                # We'll check the registry for each tool call below
            except ImportError:
                pass  # Minimax patch not available, skip

        for i, tool_call in enumerate(response.info["tool_calls"]):
            # Get reasoning for this specific tool call, or use default if available
            reasoning_content = tool_call_reasoning_map.get(
                tool_call.tool_call_id,
                tool_call_reasoning_map.get(None)  # Fallback to default (DeepSeek R1 style)
            )

            # CRITICAL: For Minimax M2.5, check the global registry if we don't have reasoning yet
            if not reasoning_content:
                try:
                    from eigent_search.minimax_m25_patch import get_minimax_content
                    minimax_data = get_minimax_content(tool_call.tool_call_id)
                    if minimax_data:
                        _, reasoning_content = minimax_data
                except (ImportError, NameError):
                    pass  # Minimax patch not available, skip

            tool_call_info = ToolCallInfo.extract_from_tool_call_record(i, tool_call, reasoning_content)
            tool_name = tool_call_info.tool_name
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            trajectory.append(tool_call_info)

        return cls(
            tool_counts=tool_counts,
            trajectory_length=len(trajectory),
            trajectory=trajectory,
        )
