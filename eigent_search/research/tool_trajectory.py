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

import ast

logger = get_logger(__name__)


class ToolCallInfo(BaseModel):
    """Information about a tool call.

    We discard the images field from `ToolCallingRecord` because it is not needed here.
    """

    tool_call_index: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any

    @classmethod
    def extract_from_tool_call_record(
        cls, tool_call_index: int, tool_call_record: ToolCallingRecord
    ) -> ToolCallInfo:
        return cls(
            tool_call_index=tool_call_index,
            tool_call_id=tool_call_record.tool_call_id,
            tool_name=tool_call_record.tool_name,
            arguments=tool_call_record.args,
            result=tool_call_record.result,
        )

    @classmethod
    def extract_from_memory_messages(
            cls, tool_call_index: int, tool_call_dict: dict, messages: list
    ) -> "ToolCallInfo":
        """从memory消息中提取单个工具调用信息"""
        import json

        name = tool_call_dict['function']['name']
        args_raw = tool_call_dict['function']['arguments']
        call_id = tool_call_dict['id']

        # 解析arguments
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw

        # try:
        #     args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        # except:
        #     args = {}

        # 找结果
        result = None
        for msg in messages:
            if msg.get('role') == 'tool' and msg.get('tool_call_id') == call_id:
                content = msg.get('content', '')
                # result = ast.literal_eval(content)

                try:
                    result = ast.literal_eval(content)
                except:
                    result = content
                break

        return cls(
            tool_call_index=tool_call_index,
            tool_call_id=call_id,
            tool_name=name,
            arguments=args,
            result=result
        )


class ToolTrajectory(BaseModel):
    tool_counts: dict[str, int]
    trajectory_length: int
    trajectory: list[ToolCallInfo]

    @classmethod
    def extract_from_response(cls, response: ChatAgentResponse) -> "ToolTrajectory":
        trajectory = []
        tool_counts = {}

        for i, tool_call in enumerate(response.info["tool_calls"]):
            tool_call_info = ToolCallInfo.extract_from_tool_call_record(i, tool_call)
            tool_name = tool_call_info.tool_name
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            trajectory.append(tool_call_info)

        return cls(
            tool_counts=tool_counts,
            trajectory_length=len(trajectory),
            trajectory=trajectory,
        )

    @classmethod
    def extract_from_memory_context(cls, memory_context) -> "ToolTrajectory":
        if not memory_context:
            return cls(tool_counts={}, trajectory_length=0, trajectory=[])

        messages, *_ = memory_context
        trajectory = []
        tool_counts = {}
        tool_call_index = 0

        for msg in messages:
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    tool_info = ToolCallInfo.extract_from_memory_messages(
                        tool_call_index, tc, messages
                    )

                    trajectory.append(tool_info)
                    tool_counts[tool_info.tool_name] = tool_counts.get(tool_info.tool_name, 0) + 1
                    tool_call_index += 1

        return cls(
            tool_counts=tool_counts,
            trajectory_length=len(trajectory),
            trajectory=trajectory,
        )