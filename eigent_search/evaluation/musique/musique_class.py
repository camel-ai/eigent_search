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

import json
from pydantic import BaseModel, Field
from typing import Literal, List
from camel.agents import ChatAgent
from datasets import load_dataset, Dataset
from ..base import BaseEvaluator, EvaluationRequest, EvaluationResult
from .answer import (
    metric_max_over_ground_truths,
    compute_exact,
    compute_f1,
    compute_acc,
    normalize_answer
)


class MusiQuePayload(BaseModel):
    """Single MuSiQue sample (only fields required for answer quality assessment, cause it also has retrieve quality assessment)"""
    qid: str = Field(..., description="The question id.")
    reference_answer: str = Field(..., description="The ground truth answer.")
    reference_answer_aliases: List[str] = Field(default_factory=list, description="The ground truth answer aliase.")
    answerable: bool = Field(True, description="is the ground truth answerable or not.")
    model_answer: str = Field(..., description="The model predicted answer.")


class MusiQueEvaluator(BaseEvaluator):
    """
    The evaluation scoring class aligned with official evaluation method for evaluating the quality of predicted answers for MuSique dataset.
    """
    @staticmethod
    def load_dataset() -> Dataset:
        return load_dataset("dgslibisey/MuSiQue")["validation"]

    def create_request(
        self,
        *,
        qid: str,
        reference_answer: str,
        reference_answer_aliases: List[str],
        answerable: bool,
        model_answer: str,
    ) -> EvaluationRequest[MusiQuePayload]:
        payload = MusiQuePayload(
            qid=qid,
            reference_answer=reference_answer,
            reference_answer_aliases=reference_answer_aliases or [],
            answerable=answerable,
            model_answer=model_answer,
        )
        return EvaluationRequest(payload=payload)

    def evaluate(self, request: EvaluationRequest[MusiQuePayload]) -> EvaluationResult:
        p = request.payload
        model_answer_json = json.loads(p.model_answer)
        model_answer = model_answer_json["answer"]
        if p.answerable:
            ground_truths = [p.reference_answer] + list(p.reference_answer_aliases or [])
            ground_truths = [gt for gt in ground_truths if gt and normalize_answer(gt).strip()] 
            em = metric_max_over_ground_truths(compute_exact, model_answer, ground_truths)
            f1 = metric_max_over_ground_truths(compute_f1, model_answer, ground_truths)
            acc = metric_max_over_ground_truths(compute_acc, model_answer, ground_truths)
        else:
            em, f1 , acc= 0.0, 0.0, 0.0 

        metrics = {"answer_em": round(float(em), 3), "answer_f1": round(float(f1), 3), "answer_acc": round(float(acc), 3)}
        return EvaluationResult(
            **request.model_dump(),
            metrics=metrics)
