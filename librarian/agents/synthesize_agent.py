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

from pydantic import BaseModel, Field
from textwrap import dedent
from camel.models import BaseModelBackend
from .struct_agent import StructAgent

# AgentOps decorator setting
try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import track_agent
    else:
        raise ImportError
except (ImportError, AttributeError):
    from camel.utils import track_agent


class SynthesisResponse(BaseModel):
    answer: str = Field(description="The answer to the complex question.")
    confidence: float = Field(description="Confidence in the answer.")
    reasoning: str = Field(
        description="Step-by-step reasoning that led to this answer."
    )
    sources: list[str] = Field(
        description="List of sources that contributed to the answer."
    )


@track_agent(name="SynthesizeAgent")
class SynthesizeAgent(StructAgent):
    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        system_message = dedent("""
        You are a synthesis agent. Your job is to combine information from multiple searches
        to answer a complex question.
        
        Given a complex question and a set of search findings, return a JSON object with the following structure:
        {
            "answer": "The answer to the complex question",
            "confidence": 0.0 to 1.0 (how confident you are in this answer),
            "reasoning": "Step-by-step reasoning that led to this answer",
            "sources": ["URL1", "URL2", ...] (list of sources that contributed to the answer)
        }
        
        Be honest about uncertainty. If you cannot answer the question with high confidence,
        explain what additional information would be needed.
        """).strip()
        super().__init__(
            response_format=SynthesisResponse,
            system_message=system_message,
            model=model,
            *args,
            **kwargs,
        )
