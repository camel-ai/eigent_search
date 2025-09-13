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
from typing import List
from ..base import BaseEvaluator, EvaluationRequest, EvaluationResult
from datasets import load_dataset, Dataset
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
    answer: str = Field(..., description="The ground truth answer.")
    answer_aliases: List[str] = Field(default_factory=list, description="The ground truth answer aliase.")
    answerable: bool = Field(True, description="is the ground truth answerable or not.")
    prediction: str = Field("", description="The predicted answer.")

class MusiQueEvaluator(BaseEvaluator):
    """
    The evaluation scoring class aligned with official evaluation method for evaluating the quality of predicted answers for MuSique dataset.
    """
    @staticmethod
    def load_dataset() -> Dataset:
        return load_dataset("dgslibisey/MuSiQue")

    def create_request(
        self,
        *,
        qid: str,
        answer: str,
        answer_aliases: List[str],
        answerable: bool,
        prediction: str,
    ) -> EvaluationRequest[MusiQuePayload]:
        payload = MusiQuePayload(
            qid=qid,
            answer=answer,
            answer_aliases=answer_aliases or [],
            answerable=answerable,
            prediction=prediction,
        )
        return EvaluationRequest(payload=payload)

    def evaluate(self, request: EvaluationRequest[MusiQuePayload]) -> EvaluationResult:
        p = request.payload

        if p.answerable:
            # ground_truths = [p.answer] + list(p.answer_aliases) 
            ground_truths = [p.answer] + list(p.answer_aliases or [])
            ground_truths = [gt for gt in ground_truths if gt and normalize_answer(gt).strip()] 
            em = metric_max_over_ground_truths(compute_exact, p.prediction, ground_truths)
            f1 = metric_max_over_ground_truths(compute_f1, p.prediction, ground_truths)
            acc = metric_max_over_ground_truths(compute_acc, p.prediction, ground_truths)
        else:
            em, f1 , acc= 0.0, 0.0, 0.0 

        metrics = {"answer_em": round(float(em), 3), "answer_f1": round(float(f1), 3), "answer_acc": round(float(acc), 3)}
        return EvaluationResult(score=float(metrics["answer_f1"]), metrics=metrics)