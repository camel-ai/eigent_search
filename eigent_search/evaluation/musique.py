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

import collections
import json
import re
import string
from typing import List

from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

from .base import BaseEvaluator, EvaluationRequest, EvaluationResult


class MusiQuePayload(BaseModel):
    """Single MuSiQue sample (only fields required for answer quality assessment, cause it also has retrieve quality assessment)"""

    qid: str = Field(..., description="The question id.")
    reference_answer: str = Field(
        ..., description="The reference (ground truth) answer."
    )
    reference_answer_aliases: List[str] = Field(
        default_factory=list, description="The reference (ground truth) answer aliases."
    )
    answerable: bool = Field(
        True,
        description="Whether the reference (ground truth) answer is answerable or not.",
    )
    model_answer: str = Field(..., description="The model-predicted answer.")


class MusiQueEvaluator(BaseEvaluator):
    """The evaluation scoring class aligned with official evaluation method for evaluating the quality of predicted answers for MuSique dataset.

    It doesn't require a judge agent.
    """

    def __init__(self):
        pass

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
            reference_answers = [p.reference_answer] + list(
                p.reference_answer_aliases or []
            )
            reference_answers = [
                ref_answer
                for ref_answer in reference_answers
                if ref_answer and self.normalize_answer(ref_answer).strip()
            ]
            max_metrics = self.compute_max_metrics(model_answer, reference_answers)
            answer_em = max_metrics["answer_em"]
            answer_f1 = max_metrics["answer_f1"]
            answer_acc = max_metrics["answer_acc"]
        else:
            answer_em, answer_f1, answer_acc = 0.0, 0.0, 0.0

        metrics = {
            "answer_em": round(float(answer_em), 3),
            "answer_f1": round(float(answer_f1), 3),
            "answer_acc": round(float(answer_acc), 3),
        }
        return EvaluationResult(**request.model_dump(), metrics=metrics)

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(answer)))).strip()

    @classmethod
    def get_normalized_tokens(cls, answer: str) -> List[str]:
        """Get the tokens of the normalized answer."""
        if not answer:
            return []
        return cls.normalize_answer(answer).split()

    @classmethod
    def compute_exact_match(cls, model_answer: str, ref_answer: str) -> int:
        """Compute the exact match between the reference and predicted answers."""
        return int(
            cls.normalize_answer(model_answer) == cls.normalize_answer(ref_answer)
        )

    @classmethod
    def compute_f1(cls, model_answer: str, ref_answer: str) -> float:
        """Compute the F1 score between the reference and predicted answers."""
        pred_tokens = cls.get_normalized_tokens(model_answer)
        ref_tokens = cls.get_normalized_tokens(ref_answer)
        common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(pred_tokens == ref_tokens)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @classmethod
    def compute_accuracy(cls, model_answer: str, ref_answer: str) -> int:
        """
        Accuracy = 1 if prediction contains any ground truth or alias (after normalize).
        """
        pred_norm = cls.normalize_answer(model_answer)
        ref_norm = cls.normalize_answer(ref_answer)
        if not ref_norm:
            return 0
        pattern = r"\b" + re.escape(ref_norm) + r"\b"
        return 1 if re.search(pattern, pred_norm) else 0

    @classmethod
    def compute_max_metrics(
        cls, model_answer: str, ref_answers: List[str]
    ) -> dict[str, float]:
        """Compute the maximum metrics between the predicted answer and all possible reference answers."""
        max_metrics = {
            "answer_em": 0.0,
            "answer_f1": 0.0,
            "answer_acc": 0.0,
        }
        for ref_answer in ref_answers:
            em = cls.compute_exact_match(model_answer, ref_answer)
            f1 = cls.compute_f1(model_answer, ref_answer)
            acc = cls.compute_accuracy(model_answer, ref_answer)
            max_metrics["answer_em"] = max(max_metrics["answer_em"], em)
            max_metrics["answer_f1"] = max(max_metrics["answer_f1"], f1)
            max_metrics["answer_acc"] = max(max_metrics["answer_acc"], acc)
        return max_metrics
