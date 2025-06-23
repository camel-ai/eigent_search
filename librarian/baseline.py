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
"""Predefined :class:`ChatAgent` subclasses for baseline agents."""

from textwrap import dedent
from pydantic import BaseModel, Field
from camel.models import BaseModelBackend
from camel.messages import BaseMessage
from camel.responses import ChatAgentResponse
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


class DirectAnswerResponse(BaseModel):
    answer: str = Field(..., description="The predicted answer.")


@track_agent(name="DirectAnswerAgent")
class DirectAnswerAgent(ChatAgent):
    r"""A :class:`ChatAgent` that outputs a direct answer with a predefined response format."""

    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        # Predefined system message for direct answering
        system_message = dedent("""
        You are a helpful assistant who answers the question directly.
        
        Final Output Format:
        ```
        Answer: ...
        ```
        """).strip()
        super().__init__(system_message=system_message, model=model, *args, **kwargs)

    def step(self, input_message: BaseMessage | str) -> ChatAgentResponse:
        return super().step(input_message, response_format=DirectAnswerResponse)


class ChainOfThoughtResponse(BaseModel):
    reasoning: str = Field(..., description="The step-by-step reasoning process.")
    answer: str = Field(..., description="The predicted answer.")


@track_agent(name="ChainOfThoughtAgent")
class ChainOfThoughtAgent(ChatAgent):
    r"""A :class:`StructAgent` that outputs an answer through step-by-step reasoning."""

    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        system_message = dedent("""
        You are a helpful assistant who reasons step by step to answer the question.
        
        Final Output Format:
        ```
        Step-by-step reasoning: ...
        Answer: ...
        ```
        """).strip()
        super().__init__(system_message=system_message, model=model, *args, **kwargs)

    def step(self, input_message: BaseMessage | str) -> ChatAgentResponse:
        return super().step(input_message, response_format=ChainOfThoughtResponse)


class KnowledgeThenReasoningResponse(BaseModel):
    knowledge: str = Field(..., description="The retrieved knowledge.")
    reasoning: str = Field(..., description="The step-by-step reasoning process.")
    answer: str = Field(..., description="The predicted answer.")


@track_agent(name="KnowledgeThenReasoningAgent")
class KnowledgeThenReasoningAgent(ChatAgent):
    r"""A :class:`StructAgent` that outputs an answer in two steps: first presenting knowledge from its memory, then reasoning step-by-step."""

    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        system_message = dedent("""
        You are a helpful assistant to answer a question in two steps.

        Step 1: Recall and list all relevant facts to the question.

        Step 2: Using only the facts from Step 1, perform a step-by-step reasoning process that leads to your final answer.  

        Final Output Format:
        ```
        Retrieved Facts: ...
        Step-by-Step Reasoning: ...
        Answer: ...
        ```
        """).strip()
        super().__init__(system_message=system_message, model=model, *args, **kwargs)

    def step(self, input_message: BaseMessage | str) -> ChatAgentResponse:
        return super().step(
            input_message, response_format=KnowledgeThenReasoningResponse
        )
