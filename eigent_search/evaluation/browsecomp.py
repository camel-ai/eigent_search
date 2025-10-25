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
import base64
import hashlib
from typing import Literal

from camel.logger import get_logger
from camel.agents import ChatAgent
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

from .base import BaseEvaluator, EvaluationRequest, EvaluationResult

logger = get_logger(__name__)

QUERY_TEMPLATE = """
{query}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


# template imported from OpenAI's Simple Eval
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {query}

[response]: {model_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {reference_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class BrowseCompPayload(BaseModel):
    query: str = Field(..., description="The question to evaluate.")
    reference_answer: str = Field(..., description="The ground truth answer.")
    model_answer: str = Field(..., description="The predicted answer.")


class BrowseCompGrade(BaseModel):
    """The grade of the predicted answer for BrowseComp."""

    grade: Literal["yes", "no"] = Field(
        ..., description="The grade of the predicted answer."
    )


class BrowseCompEvaluator(BaseEvaluator):
    """A chat agent-based class for evaluating the quality of predicted answers for BrowseComp."""

    def __init__(self, judge_agent: ChatAgent):
        self.judge_agent = judge_agent

    @staticmethod
    def load_dataset() -> Dataset:
        test_samples = load_dataset("smolagents/browse_comp")["test"]
        decoded_samples = []
        for test_sample in test_samples:
            # There shouldn't be any missing canary, and decrypt should never fail
            canary = test_sample["canary"]
            problem = test_sample["problem"]
            answer = test_sample["answer"]
            test_sample["problem"] = BrowseCompEvaluator.decrypt(problem, canary)
            test_sample["answer"] = BrowseCompEvaluator.decrypt(answer, canary)
            decoded_samples.append(test_sample)
        return Dataset.from_list(decoded_samples)

    def create_request(
        self, query: str, reference_answer: str, model_answer: str
    ) -> EvaluationRequest[BrowseCompPayload]:
        return EvaluationRequest(
            payload=BrowseCompPayload(
                query=query,
                reference_answer=reference_answer,
                model_answer=model_answer,
            )
        )

    def evaluate(
        self, request: EvaluationRequest[BrowseCompPayload]
    ) -> EvaluationResult:
        self.judge_agent.reset()
        response = self.judge_agent.step(
            GRADER_TEMPLATE.format(
                query=QUERY_TEMPLATE.format(query=request.payload.query),
                reference_answer=request.payload.reference_answer,
                model_answer=request.payload.model_answer,
            ),
            response_format=BrowseCompGrade,
        )
        grade = eval(response.msgs[0].content.strip())["grade"]
        return EvaluationResult(
            **request.model_dump(),
            score=1.0 if grade == "yes" else 0.0,
            metrics={"grade": grade},
        )

    @staticmethod
    def derive_key(password: str, length: int) -> bytes:
        """Derive a fixed-length key from the password using SHA256."""
        hasher = hashlib.sha256()
        hasher.update(password.encode())
        key = hasher.digest()
        return key * (length // len(key)) + key[: length % len(key)]

    @staticmethod
    def decrypt(ciphertext_b64: str, password: str) -> str:
        """Decrypt base64-encoded ciphertext with XOR."""
        encrypted = base64.b64decode(ciphertext_b64)
        key = BrowseCompEvaluator.derive_key(password, len(encrypted))
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
        return decrypted.decode()
