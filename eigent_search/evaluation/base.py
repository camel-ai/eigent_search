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
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pydantic import BaseModel, Field


BenchmarkPayload = TypeVar("T", bound=BaseModel)


class EvaluationRequest(BaseModel, Generic[BenchmarkPayload]):
    payload: BenchmarkPayload
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(EvaluationRequest[BenchmarkPayload]):
    """Result of an evaluation with an overall score and detailed metrics, along with the original request payload and metadata."""

    score: float = Field(
        default_factory=float, description="Standard score for the evaluation."
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional detailed metrics for the evaluation.",
    )


class BaseEvaluator(ABC):
    """Abstract interface for scoring agent responses."""

    @abstractmethod
    def load_dataset(
        self, *args, **kwargs
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """Load the dataset for this benchmark."""
        ...

    @abstractmethod
    def create_request(self, *args, **kwargs) -> EvaluationRequest[BenchmarkPayload]:
        """Create an :obj:`EvaluationRequest` from a :obj:`BenchmarkPayload`."""
        ...

    @abstractmethod
    def evaluate(
        self, request: EvaluationRequest[BenchmarkPayload]
    ) -> EvaluationResult:
        """Compute a scalar score and per-metric breakdown for a given :obj:`EvaluationRequest`."""
        ...
