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
    
    
class PlanResponse(BaseModel):
    analysis: str = Field(description="Brief analysis of the question and what makes it complex.")
    search_plan: list[str] = Field(description="List of search queries and their purposes.")


@track_agent(name="PlanAgent")
class PlanAgent(StructAgent):
    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        system_message = dedent("""
        You are a search planner. Your job is to break down complex questions into a series of 
        simpler search queries that can be used to find the answer.
        
        For each complex question, return a JSON object with the following structure:
        ```
        {
            "analysis": "Brief analysis of the question and what makes it complex",
            "search_plan": [
                {
                    "query": "First search query",
                    "purpose": "What we hope to learn from this query"
                },
                {
                    "query": "Second search query",
                    "purpose": "What we hope to learn from this query"
                },
                ...
            ]
        }
        ```
        
        Make your queries specific and targeted. Focus on finding factual information.
        """).strip()
        super().__init__(
            response_format=PlanResponse,
            system_message=system_message,
            model=model,
            *args,
            **kwargs,
        )