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
from enum import Enum
import itertools
import json
import os
from pathlib import Path
from typing import Type

from camel.agents import ChatAgent
from camel.logger import get_logger
from camel.models import BaseModelBackend, ModelFactory
from camel.toolkits import BaseToolkit, RegisteredAgentToolkit
from camel.types import ModelType
from camel.types import ModelPlatformType
from dotenv import load_dotenv
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

# Load environment variables from .env file
load_dotenv()


def get_required_env(key: str) -> str:
    """Get environment variable or raise an exception if not set."""
    value = os.getenv(key)
    if value is None:
        raise ValueError(
            f"Missing required environment variable: {key}. "
            f"Please set it in your .env file or environment."
        )
    return value


DEFAULT_MAX_ORCHESTRATOR_RETRIES = 5
DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP = 5


class SearchAgentType(Enum):
    """Predefined agent types for using different tools in the search environment."""

    # Base agent with no tools (model only)
    BASE = "base"
    # Search agent only using search_google tool
    SEARCH_ONLY = "search_only"
    # Default search agent from Eigent
    EIGENT_SEARCH = "eigent_search"
    # Eigent search agent enhanced with query processing tools
    EIGENT_SEARCH_Q_PLUS = "eigent_search_q+"

    # Customized agent (no preset system prompt or tools)
    CUSTOMIZED = "customized"


class BackendModelConfig(Enum):
    """Backend model configurations for different models.

    For Azure models:
        - url: reads from AZURE_OPENAI_BASE_URL env var
        - api_key: reads from AZURE_OPENAI_API_KEY env var
        - api_version: Azure API version (e.g., "2024-12-01-preview")

    For OpenAI models:
        - api_key: reads from OPENAI_API_KEY env var (handled by CAMEL internally)
    """

    # Azure models
    AZURE_GPT_5_MINI = {
        "model_type": "gpt-5-mini",
        "model_platform": ModelPlatformType.AZURE,
        "temperature": 1.0,  # must be 1.0 for GPT-5-mini
        "api_version": "2024-12-01-preview",
    }
    AZURE_GPT_4_1 = {
        "model_type": "gpt-4.1",
        "model_platform": ModelPlatformType.AZURE,
        "temperature": 0.0,
        "api_version": "2024-12-01-preview",
    }
    AZURE_GPT_4_1_MINI = {
        "model_type": "gpt-4.1-mini",
        "model_platform": ModelPlatformType.AZURE,
        "temperature": 0.0,
        "api_version": "2024-12-01-preview",
    }
    AZURE_GPT_4_O= {
        "model_type": "gpt-4o",
        "model_platform": ModelPlatformType.AZURE,
        "temperature": 0.0,
        "api_version": "2024-12-01-preview",
    }

    # OpenAI models
    GPT_5_MINI = {
        "model_type": ModelType.GPT_5_MINI,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 1.0,  # must be 1.0 for GPT-5-mini
    }
    GPT_4_1 = {
        "model_type": ModelType.GPT_4_1,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
    }
    GPT_4_1_MINI = {
        "model_type": ModelType.GPT_4_1_MINI,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
    }
    GPT_4O = {
        "model_type": ModelType.GPT_4O,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
    }
    GPT_4O_MINI = {
        "model_type": ModelType.GPT_4O_MINI,
        "model_platform": ModelPlatformType.OPENAI,
        "temperature": 0.0,
    }


class SearchConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Path Config
    working_directory: DirectoryPath = Field(default_factory=lambda: Path(os.getcwd()))

    # Model Config
    system_prompt: str | None = None
    model_type: str | ModelType = ModelType.GPT_4_1_MINI
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI
    temperature: float = 0.0
    # Azure-specific: API version (e.g., "2024-12-01-preview")
    api_version: str | None = None

    # Tool Config
    toolkits: list[BaseToolkit] = Field(default_factory=list)
    toolkits_to_register_agent: list[RegisteredAgentToolkit] = Field(
        default_factory=list
    )
    toolkits_to_cleanup: list[CleanupToolkit] = Field(default_factory=list)
    # Required environment variables for toolkits (validated in create_agent)
    required_env_vars: list[str] = Field(default_factory=list)

    # Orchestrator Config
    agent_type: SearchAgentType = SearchAgentType.CUSTOMIZED
    max_orchestrator_retries: int = DEFAULT_MAX_ORCHESTRATOR_RETRIES
    timeout_minutes_per_orchestrator_step: int = (
        DEFAULT_TIMEOUT_MINUTES_PER_ORCHESTRATOR_STEP
    )
    response_format: Type[BaseModel] | None = None

    @model_validator(mode="after")
    def _finalize(self) -> SearchConfig:
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
        # Build kwargs for ModelFactory.create
        kwargs = {
            "model_platform": self.model_platform,
            "model_type": self.model_type,
            "model_config_dict": {"temperature": self.temperature},
        }

        if self.model_platform == ModelPlatformType.AZURE:
            kwargs["url"] = get_required_env("AZURE_OPENAI_BASE_URL")
            kwargs["api_key"] = get_required_env("AZURE_OPENAI_API_KEY")
            kwargs["api_version"] = self.api_version
        elif self.model_platform == ModelPlatformType.OPENAI:
            kwargs["api_key"] = get_required_env("OPENAI_API_KEY")

        return ModelFactory.create(**kwargs)

    def create_agent(self) -> ChatAgent:
        # Validate required environment variables for toolkits
        for env_var in self.required_env_vars:
            get_required_env(env_var)

        return ChatAgent(
            system_message=self.system_prompt,
            model=self.create_model(),
            tools=list(
                itertools.chain.from_iterable(
                    toolkit.get_tools() for toolkit in self.toolkits
                )
            ),
            toolkits_to_register_agent=self.toolkits_to_register_agent,
            summarize_threshold=None,
        )

    def set_preset_system_prompt(self, agent_type: SearchAgentType):
        self.system_prompt = prompt_manager.get_preset_system_prompt(
            self.working_directory.as_posix(), agent_type
        )

    def set_preset_tools(self, agent_type: SearchAgentType):
        # Base agent: no tools, no required env vars
        if agent_type == SearchAgentType.BASE:
            self.toolkits = []
            self.toolkits_to_register_agent = []
            self.toolkits_to_cleanup = []
            self.required_env_vars = []
            return

        # All other preset agent types require Google Search API keys
        eigent_search_toolkit = EigentSearchToolkit(
            working_directory=self.working_directory,
            exclude_search_domains=["huggingface.co", "hf.co", "oxen.ai"],
        )
        self.required_env_vars = ["GOOGLE_API_KEY", "SEARCH_ENGINE_ID"]

        if agent_type == SearchAgentType.SEARCH_ONLY:
            self.toolkits = [eigent_search_toolkit.search_toolkit]
            self.toolkits_to_register_agent = []

        elif agent_type == SearchAgentType.EIGENT_SEARCH:
            self.toolkits = [eigent_search_toolkit]
            self.toolkits_to_register_agent = [eigent_search_toolkit.browser_toolkit]
            self.toolkits_to_cleanup = [eigent_search_toolkit]

        elif agent_type == SearchAgentType.EIGENT_SEARCH_Q_PLUS:
            query_processing_toolkit = QueryProcessingToolkit(
                exclude_domains=["huggingface.co", "hf.co", "oxen.ai"],
            )
            eigent_search_toolkit.register(query_processing_toolkit)
            eigent_search_toolkit.search_toolkit.get_tools = lambda: []
            self.toolkits = [eigent_search_toolkit, query_processing_toolkit]
            self.toolkits_to_register_agent = [eigent_search_toolkit.browser_toolkit]
            self.toolkits_to_cleanup = [eigent_search_toolkit]

    def _serialize_value(self, value):
        """Recursively serialize values to make them JSON-compatible."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, "as_posix"):  # Path-like objects
            return value.as_posix()
        elif hasattr(value, "value"):  # Enum with value attribute
            return value.value
        elif hasattr(value, "__name__"):  # Type/class objects
            return value.__name__
        elif hasattr(value, "__class__"):  # Other objects
            return value.__class__.__name__
        else:
            return str(value)

    def model_dump_json(self, indent: int = 2, **kwargs) -> str:
        """Override model_dump_json to handle unserializable attributes."""
        # Get the default model dump
        model_dict = self.model_dump(**kwargs)

        # Serialize all values to make them JSON-compatible
        serialized_dict = {}
        for key, value in model_dict.items():
            serialized_dict[key] = self._serialize_value(value)

        # Convert to JSON string
        return json.dumps(serialized_dict, indent=indent)

    def to_json_dict(self) -> dict:
        """Get a JSON-serializable dictionary representation of the config."""
        model_dict = self.model_dump()
        return {key: self._serialize_value(value) for key, value in model_dict.items()}


class LLMasJudgeConfig(BaseModel):
    model_type: str | ModelType = ModelType.GPT_4_1
    model_platform: ModelPlatformType = ModelPlatformType.OPENAI
    temperature: float = 0.0
    # Azure-specific: API version (e.g., "2024-12-01-preview")
    api_version: str | None = None

    def create_model(self) -> BaseModelBackend:
        # Build kwargs for ModelFactory.create
        kwargs = {
            "model_platform": self.model_platform,
            "model_type": self.model_type,
            "model_config_dict": {"temperature": self.temperature},
        }

        if self.model_platform == ModelPlatformType.AZURE:
            kwargs["url"] = get_required_env("AZURE_OPENAI_BASE_URL")
            kwargs["api_key"] = get_required_env("AZURE_OPENAI_API_KEY")
            kwargs["api_version"] = self.api_version
        elif self.model_platform == ModelPlatformType.OPENAI:
            kwargs["api_key"] = get_required_env("OPENAI_API_KEY")

        return ModelFactory.create(**kwargs)

    def create_agent(self) -> ChatAgent:
        return ChatAgent(model=self.create_model(), summarize_threshold=None)
