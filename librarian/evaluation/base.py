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
from typing import Any
from pydantic import BaseModel, Field


class EvaluationRequest(BaseModel):
    """Container for inputs to evaluators."""
    context: dict[str, Any]

class EvaluationResult(BaseModel):
    """Result of an evaluation with an overall score and detailed metrics."""
    score: float = Field(default_factory=float, description="Standard score for the evaluation.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Optional detailed metrics for the evaluation.")

class BaseEvaluator(ABC):
    """Abstract interface for scoring agent responses."""

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Compute a scalar score and per-metric breakdown for a given :obj:`EvaluationRequest`."""
        ...

    def __call__(self, requests: list[EvaluationRequest]) -> list[EvaluationResult]:
        return [self.evaluate(request) for request in requests]
