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


class SearchResponse(BaseModel):
    query: str = Field(description="The search query that was used.")
    findings: list[dict[str, str]] = Field(
        description="List of search findings including fact, source and confidence."
    )


@track_agent(name="SearchAgent")
class SearchAgent(StructAgent):
    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        system_message = dedent("""
        You are a search agent. Your job is to perform web searches and extract the most relevant 
        information from the search results.
        
        For each search query, return a JSON object with the following structure:
        ```
        {
            "query": "The search query that was used",
            "findings": [
                {
                    "fact": "A relevant fact extracted from the search results",
                    "source": "The URL where this fact was found",
                    "confidence": 0.0 to 1.0 (how confident you are in this fact)
                },
                ...
            ]
        }
        ```
        
        Focus on extracting factual information that is directly relevant to the original question.
        """).strip()
        super().__init__(
            response_format=SearchResponse,
            system_message=system_message,
            model=model,
            *args,
            **kwargs,
        )
