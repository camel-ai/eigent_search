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


import abc
from typing import Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from camel.models import BaseModelBackend
from camel.agents.chat_agent import ChatAgent
from camel.toolkits import FunctionTool
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


# TODO: need to revise
class ResearchResult(BaseModel):
    summary: str = Field(..., description="The summary of the research results.")
    sources: List[str] = Field(..., description="The sources of the research results.")


@track_agent(name="BaseResearchAgent")
class BaseResearchAgent(abc.ABC, ChatAgent):
    """Base class for both lead and sub research agents."""

    role: str = "generic"

    def __init__(
        self,
        system_message: str,
        model: BaseModelBackend,
        research_toolkit: QueryProcessingToolkit,
        parent: Optional["LeadResearcher"] = None,
        *args,
        **kwargs,
    ) -> None:
        self.research_toolkit = research_toolkit
        self.parent = parent
        super(ChatAgent, self).__init__(
            system_message=system_message,
            model=model,
            tools=self.research_toolkit.get_tools(),
            *args,
            **kwargs,
        )

    @abc.abstractmethod
    def research(self, query: str) -> Dict[str, Any]:
        r"""Research `query` and return an answer to the query grounded on evidence."""
        ...

    @abc.abstractmethod
    def reflect(self, research_results: List[Dict[str, Any]]) -> str:
        r"""Reflect on the findings from the research results."""
        ...

    @abc.abstractmethod
    def complete_task(self, reflection: str) -> bool:
        r"""Based on the reflection, determine if the research task is complete."""
        ...

    @property
    def tools(self) -> List[FunctionTool]:  # noqa: D401
        return self.research_toolkit.get_tools()


@track_agent(name="LeadResearcher")
class LeadResearcher(abc.ABC, BaseResearchAgent):
    """Plans the work, spawns sub‑agents, aggregates results."""

    role = "leader"

    @abc.abstractmethod
    def create_research_plan(self, query: str) -> Dict[str, Any]:
        r"""Create a research plan for the given query."""
        ...

    @abc.abstractmethod
    def assign_task(self, sub_query: str) -> Dict[str, Any]:
        r"""Assign a task to a spawned sub-research agent (`JuniorResearcher`) and return the research results."""
        ...

    def research(self, query: str) -> Dict[str, Any]:  # noqa: D401
        plan = self.create_research_plan(query)
        sub_tasks: Sequence[str] = plan.get("sub_tasks", [])  # type: ignore[arg-type]

        # TODO: need to revise async implementation
        research_results: List[Dict[str, Any]] = []
        for task in sub_tasks:
            sub_research_results = self.assign_task(task)
            research_results.append(sub_research_results)

        reflection = self.reflect(research_results)
        if not self.complete_task(reflection):
            return self.research(reflection.refined_query)

        return {
            "query": query,
            "answer": reflection.answer,
            "evidence": reflection.evidence,
        }


@track_agent(name="JuniorResearcher")
class JuniorResearcher(BaseResearchAgent):
    """Runs exactly one sub‑query, then stops."""

    role = "worker"

    def research(self, query: str) -> Dict[str, Any]:
        r"""Research `query` and return an answer to the query grounded on evidence.

        Junior researcher always stops after first pass to avoid cycles.
        """
        ...
