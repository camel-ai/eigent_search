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
from camel.logger import get_logger
from camel.agents import ChatAgent
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field
from typing import Literal
from .base import BaseEvaluator, EvaluationRequest, EvaluationResult
import base64

logger = get_logger(__name__)

# System and answer format prompts
QUESTION_TEMPLATE = """
你是一个通用人工智能助手。我将向你提出一个学术问题, 请尽可能简洁地给出解题思路, 并用以下模版作为回答的结尾:

最终答案:[你的答案]

不要在最终答案周围添加任何多余的符号, 不要使用换行（在同一行中完成回答）
""".strip()

OBJECTIVE_PROMPT = "对于本题, 你的答案必须是尽可能简洁的数值, 短语, 或者数学表达式; 如果答案有多个, 使用逗号将它们隔开。"
CHOICE_PROMPT = "对于本题, 选出所有符合的选项, 少选、多选或错选都不得分; 如果选项有多个, 连续列出所有选项, 不要使用逗号或空格分隔。"

# template imported from https://github.com/xbench-ai/xbench-evals
GRADER_TEMPLATE = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


class XbenchPayload(BaseModel):
    query: str = Field(..., description="The question to evaluate.")
    reference_answer: str = Field(..., description="The ground truth answer.")
    model_answer: str = Field(..., description="The predicted answer.")


class XbenchGrade(BaseModel):
    """The grade of the predicted answer for Xbench."""

    grade: Literal["正确", "错误"] = Field(
        ..., description="The grade of the predicted answer."
    )


class XbenchEvaluator(BaseEvaluator):
    """A chat agent-based class for evaluating the quality of predicted answers for Xbench."""

    def __init__(self, judge_agent: ChatAgent):
        self.judge_agent = judge_agent

    @staticmethod
    def load_dataset() -> Dataset:
        test_samples = load_dataset("xbench/DeepSearch")["train"]
        decoded_samples = []
        for test_sample in test_samples:
            # There shouldn't be any missing canary, and decrypt should never fail
            test_sample["original_id"] = str(test_sample.pop("id"))
            canary = test_sample["canary"]
            problem = test_sample["prompt"]
            problem = XbenchEvaluator.xor_decrypt(base64.b64decode(problem), canary).decode('utf-8')
            question_type = test_sample["type"] if "type" in test_sample else "问答题"
            answer = test_sample["answer"]
            test_sample["problem"] = XbenchEvaluator.get_question_prompt(problem, question_type)
            test_sample["answer"] = XbenchEvaluator.xor_decrypt(base64.b64decode(answer), canary).decode('utf-8')
            decoded_samples.append(test_sample)
        return Dataset.from_list(decoded_samples)

    def create_request(
        self, query: str, reference_answer: str, model_answer: str
    ) -> EvaluationRequest[XbenchPayload]:
        return EvaluationRequest(
            payload=XbenchPayload(
                query=query,
                reference_answer=reference_answer,
                model_answer=model_answer,
            )
        )

    def evaluate(
        self, request: EvaluationRequest[XbenchPayload]
    ) -> EvaluationResult:
        self.judge_agent.reset()
        response = self.judge_agent.step(
            GRADER_TEMPLATE.format(
                question=request.payload.query,
                correct_answer=request.payload.reference_answer,
                response=request.payload.model_answer,
            ),
            response_format=XbenchGrade,
        )
        grade = eval(response.msgs[0].content.strip())["grade"]
        return EvaluationResult(
            **request.model_dump(),
            score=1.0 if grade == "正确" else 0.0,
            metrics={"grade": grade},
        )

    @staticmethod
    def xor_decrypt(data, key):
        """
        XOR decrypt data with a key
        """
        key_bytes = key.encode('utf-8')
        key_length = len(key_bytes)
        return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])
    
    
    def get_question_prompt(question, question_type):
        full_prompt = QUESTION_TEMPLATE 

        match question_type:
            case "问答题":
                full_prompt += OBJECTIVE_PROMPT + "\n\n"
            case "选择题":
                full_prompt += CHOICE_PROMPT + "\n\n"
            case _:
                pass

        full_prompt += "[问题]: " + question

        return full_prompt