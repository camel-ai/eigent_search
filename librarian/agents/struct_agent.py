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

from typing import Type
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
from pydantic import BaseModel

from camel.models import BaseModelBackend
from camel.agents.chat_agent import ChatAgent

# AgentOps decorator setting
try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent


@track_agent(name="StructAgent")
class StructAgent(ChatAgent):
    r"""A :class:`ChatAgent` that must have a response format for structured
    output."""

    def __init__(
        self,
        response_format: Type[BaseModel],
        system_message: BaseMessage | str | None = None,
        model: BaseModelBackend | list[BaseModelBackend] | None = None,
        *args,
        **kwargs
    ):
        super().__init__(system_message=system_message, model=model, *args, **kwargs)
        self.response_format = response_format
        
    def step(self, input_message: BaseMessage | str) -> ChatAgentResponse:
        return super().step(input_message, response_format=self.response_format)
