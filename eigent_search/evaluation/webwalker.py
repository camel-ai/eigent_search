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
from typing import Literal
from camel.agents import ChatAgent
from datasets import load_dataset, Dataset
from .base import BaseEvaluator, EvaluationRequest, EvaluationResult


# template imported from WebWalkerQA official code
GRADER_TEMPLATE = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
""".strip()


class WebWalkerPayload(BaseModel):
    query: str = Field(..., description="The question to evaluate.")
    reference_answer: str = Field(..., description="The ground truth answer.")
    model_answer: str = Field(..., description="The predicted answer.")


class WebWalkerGrade(BaseModel):
    """The grade of the predicted answer for BrowseComp."""

    grade: Literal["Correct", "Incorrect"] = Field(
        ..., description="The grade of the predicted answer."
    )


class WebWalkerEvaluator(BaseEvaluator):
    """A chat agent-based class for evaluating the quality of predicted answers for BrowseComp."""

    def __init__(self, judge_agent: ChatAgent):
        self.judge_agent = judge_agent

    @staticmethod
    def load_dataset() -> Dataset:
        return load_dataset("callanwu/WebWalkerQA")["main"]

    def create_request(
        self, query: str, reference_answer: str, model_answer: str
    ) -> EvaluationRequest[WebWalkerPayload]:
        return EvaluationRequest(
            payload=WebWalkerPayload(
                query=query,
                reference_answer=reference_answer,
                model_answer=model_answer,
            )
        )

    def evaluate(
        self, request: EvaluationRequest[WebWalkerPayload]
    ) -> EvaluationResult:
        self.judge_agent.reset()
        response = self.judge_agent.step(
            GRADER_TEMPLATE.format(
                question=request.payload.query,
                correct_answer=request.payload.reference_answer,
                response=request.payload.model_answer,
            ),
            response_format=WebWalkerGrade,
        )
        grade = eval(response.msgs[0].content.strip())["grade"]
        return EvaluationResult(
            **request.model_dump(),
            score=1.0 if grade == "Correct" else 0.0,
            metrics={"grade": grade},
        )
