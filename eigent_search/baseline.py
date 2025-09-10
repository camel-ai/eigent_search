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
from typing import Any

from camel.agents.chat_agent import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse
from camel.toolkits import FunctionTool, SearchToolkit
from pydantic import BaseModel, Field

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


class ResearchResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )


@track_agent(name="SimpleResearchAgent")
class SimpleResearchAgent(ChatAgent):
    r"""A :class:`ChatAgent` that outputs a direct answer with a predefined response format. It has access to google search tools."""

    def __init__(self, model: BaseModelBackend, *args, **kwargs):
        # Predefined system message for direct answering
        system_message = dedent("""
        You are a helpful assistant who conducts deep research on a given question.
        
        Final Output Format:
        ```
        Answer: ...
        Search Results: ...
        ```
        """).strip()
        super().__init__(
            system_message=system_message,
            model=model,
            tools=[FunctionTool(SearchToolkit().search_google)],
            *args,
            **kwargs,
        )

    def search_google(
        self,
        query: str,
        number_of_result_pages: int = 5,
    ) -> list[dict[str, Any]]:
        r"""Use Google search engine to search information for the given query.

        Args:
            query (str): The query to be searched.
            number_of_result_pages (int): The number of result pages to
                retrieve. Adjust this based on your task - use fewer results
                for focused searches and more for comprehensive searches.
                (default: :obj:`5`)

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary
            represents a search result.

                For web search, each dictionary contains:
                - 'result_id': A number in order.
                - 'title': The title of the website.
                - 'description': A brief description of the website.
                - 'long_description': More detail of the website.
                - 'url': The URL of the website.

                Example web result:
                {
                    'result_id': 1,
                    'title': 'OpenAI',
                    'description': 'An organization focused on ensuring that
                    artificial general intelligence benefits all of humanity.',
                    'long_description': 'OpenAI is a non-profit artificial
                    intelligence research company. Our goal is to advance
                    digital intelligence in the way that is most likely to
                    benefit humanity as a whole',
                    'url': 'https://www.openai.com'
                }

        """
        query += " -site:huggingface.co"
        return SearchToolkit().search_google(
            query,
            search_type="web",
            number_of_result_pages=number_of_result_pages,
            start_page=1,
        )

    def step(self, input_message: str) -> ChatAgentResponse:
        return super().step(input_message, response_format=ResearchResponse)
        # return super().step( input_message + " -site:huggingface.co", response_format=ResearchResponse)
