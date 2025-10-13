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
import datetime
from enum import Enum
import itertools
import os
from pathlib import Path
from typing import Type

from camel.agents import ChatAgent
from camel.logger import get_logger
from camel.models import BaseModelBackend, ModelFactory
from camel.toolkits import BaseToolkit, RegisteredAgentToolkit
from camel.types import ModelType
from camel.types import ModelPlatformType
from pydantic import DirectoryPath, model_validator
from pydantic import Field
from pydantic import BaseModel

from eigent_search.prompt.prompt_manager import prompt_manager
from eigent_search.toolkit import (
    CleanupToolkit,
    EigentSearchToolkit,
    QueryProcessingToolkit,
)

logger = get_logger(__name__)

DEFAULT_MAX_ORCHESTRATOR_RETRIES = 5
DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP = 5


class SearchAgentType(Enum):
    """Predefined agent types for using different tools in the search environment."""

    # Default search agent from Eigent
    EIGENT_SEARCH = "eigent_search"
    # Eigent search agent enhanced with query processing tools
    EIGENT_SEARCH_PLUS = "eigent_search_plus"
    # Search agent only using search_google tool
    SEARCH_ONLY = "search_only"

    # Customized agent (no preset system prompt or tools)
    CUSTOMIZED = "customized"


class SearchModelConfig(Enum):
    GPT_4_1_MINI = {
        "model_type": ModelType.GPT_4_1_MINI,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
        "self_host_url": None,
        "max_orchestrator_retries": DEFAULT_MAX_ORCHESTRATOR_RETRIES,
        "timeout_minutes_per_orchestrator_step": DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP,
    }
    GPT_4O_MINI = {
        "model_type": ModelType.GPT_4O_MINI,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
        "self_host_url": None,
        "max_orchestrator_retries": DEFAULT_MAX_ORCHESTRATOR_RETRIES,
        "timeout_minutes_per_orchestrator_step": DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP,
    }

    GPT_OSS = {
        "model_type": "gpt-oss:120b",
        "model_platform": ModelPlatformType.OLLAMA,
        "temperature": 0.0,
        "self_host_url": "http://129.212.188.6:7861/v1",  # need to changed by @wendong when needed
        "max_orchestrator_retries": DEFAULT_MAX_ORCHESTRATOR_RETRIES,
        "timeout_minutes_per_orchestrator_step": DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP,
    }
    ALL = [GPT_4_1_MINI, GPT_4O_MINI, GPT_OSS]


class SearchConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Path Config
    working_directory: DirectoryPath = Field(default_factory=lambda: Path(os.getcwd()))

    # Model Config
    system_prompt: str | None = None
    model_type: str | ModelType = ModelType.GPT_4_1_MINI
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI
    temperature: float = 0.0
    self_host_url: str | None = None

    # Tool Config
    toolkits: list[BaseToolkit] = Field(default_factory=list)
    toolkits_to_register_agent: list[RegisteredAgentToolkit] = Field(
        default_factory=list
    )
    toolkits_to_cleanup: list[CleanupToolkit] = Field(default_factory=list)
    prune_tool_calls_from_memory: bool = True

    # Orchestrator Config
    agent_type: SearchAgentType = SearchAgentType.CUSTOMIZED
    max_orchestrator_retries: int = DEFAULT_MAX_ORCHESTRATOR_RETRIES
    timeout_minutes_per_orchestrator_step: int = (
        DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP
    )
    response_format: Type[BaseModel] | None = None

    @model_validator(mode="after")
    def _finalize(self) -> SearchConfig:
        # Create a timestamped working directory for the search agent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.working_directory = (
            Path(self.working_directory) / f"eigent_search_{timestamp}"
        )
        self.working_directory.mkdir(parents=True, exist_ok=True)
        if not self.agent_type == SearchAgentType.CUSTOMIZED:
            logger.info(
                f"Overriding system prompt and tools for {self.agent_type} with preset; "
                "if not desired, please use `SearchAgentType.CUSTOMIZED` to create a customized agent."
            )
            self.set_preset_system_prompt(self.agent_type)
            self.set_preset_tools(self.agent_type)
        return self

    def create_model(self) -> BaseModelBackend:
        return ModelFactory.create(
            model_platform=self.model_platform,
            model_type=self.model_type,
            model_config_dict={"temperature": self.temperature},
            url=self.self_host_url,
        )

    def create_agent(self) -> ChatAgent:
        return ChatAgent(
            system_message=self.system_prompt,
            model=self.create_model(),
            tools=list(
                itertools.chain.from_iterable(
                    toolkit.get_tools() for toolkit in self.toolkits
                )
            ),
            toolkits_to_register_agent=self.toolkits_to_register_agent,
            prune_tool_calls_from_memory=self.prune_tool_calls_from_memory,
        )

    def set_preset_system_prompt(self, agent_type: SearchAgentType):
        self.system_prompt = prompt_manager.get_preset_system_prompt(
            self.working_directory.as_posix(), agent_type
        )

    def set_preset_tools(self, agent_type: SearchAgentType):
        eigent_search_toolkit = EigentSearchToolkit(
            working_directory=self.working_directory,
            exclude_search_domains=["huggingface.co", "hf.co", "oxen.ai"],
        )

        if agent_type == SearchAgentType.SEARCH_ONLY:
            self.toolkits = [eigent_search_toolkit.search_toolkit]
            self.toolkits_to_register_agent = []

        if agent_type == SearchAgentType.EIGENT_SEARCH:
            self.toolkits = [eigent_search_toolkit]
            self.toolkits_to_register_agent = [eigent_search_toolkit.browser_toolkit]
            self.toolkits_to_cleanup = [eigent_search_toolkit]

        if agent_type == SearchAgentType.EIGENT_SEARCH_PLUS:
            query_processing_toolkit = QueryProcessingToolkit(
                exclude_domains=["huggingface.co", "hf.co", "oxen.ai"],
            )
            eigent_search_toolkit.register(query_processing_toolkit)
            eigent_search_toolkit.search_toolkit.get_tools = lambda: []
            self.toolkits = [eigent_search_toolkit, query_processing_toolkit]
            self.toolkits_to_register_agent = [eigent_search_toolkit.browser_toolkit]
            self.toolkits_to_cleanup = [eigent_search_toolkit]
