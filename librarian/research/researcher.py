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

from textwrap import dedent
from pydantic import BaseModel, Field
from camel.models import BaseModelBackend
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from camel.agents.chat_agent import ChatAgent

from .query_toolkit import QueryProcessingToolkit

# AgentOps decorator setting
try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    evidence: list[dict[str, str]] = Field(..., description="The evidence of the research results.")


@track_agent(name="ResearchAgent")
class ResearchAgent(ChatAgent):
    r"""A :class:`ChatAgent` that conducts deep research on a given question."""

    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        # Predefined system message for direct answering
        system_message = dedent("""
        You are a helpful assistant who conducts deep research on a given question.
        
        Final Output Format:
        ```
        Answer: ...
        Evidence: ...
        ```
        """).strip()
        super().__init__(system_message=system_message, model=model, tools=[QueryProcessingToolkit().get_tools()], *args, **kwargs)

    def step(self, input_message: BaseMessage | str) -> ChatAgentResponse:
        return super().step(input_message, response_format=ResearchResponse)
