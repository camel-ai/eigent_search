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

import os
import time
import json
import click
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm.auto import tqdm
from datasets import load_dataset
from typing import Annotated
from types import MethodType
from pydantic import BaseModel, Field, StringConstraints

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent
from eigent_search.evaluation import SimpleQAEvaluator

from eigent_search.evaluation.musique.musique import MusiQueEvaluator
from eigent_search.baseline import (
    DirectAnswerAgent,
    ChainOfThoughtAgent,
    KnowledgeThenReasoningAgent,
    SimpleResearchAgent,
)
from eigent_search.research import deep_search_agent_factory
from eigent_search.utils import run_agent_with_retry



TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WORKING_DIRECTORY = Path(
    os.getcwd(),
    "results",
    f"eigent_search_{TIMESTAMP}",
)
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

AGENTS = {
    "simple_research": SimpleResearchAgent,
    "direct_answer": DirectAnswerAgent,
    "chain_of_thought": ChainOfThoughtAgent,
    "knowledge_then_reasoning": KnowledgeThenReasoningAgent,
    "deep_search": lambda model: deep_search_agent_factory(model, WORKING_DIRECTORY),
}

MODEL_NAMES = {
    "gpt-4o-mini": ModelType.GPT_4O_MINI,
    "gpt-4.1-mini": ModelType.GPT_4_1_MINI,
    "gpt-oss": "gpt-oss:120b",  # Example Ollama model
}



class MusiQueResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: list[str] = Field(
        ..., description="The search results that lead to the answer."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(WORKING_DIRECTORY / "musique_eval.log"),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger("__name__")


@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), default="deep_search") # origin: required=True
@click.option("--model_name", "-m", type=click.Choice(MODEL_NAMES.keys()), default="gpt-4.1-mini")
@click.option("--num_questions", "-n", type=int, default=5)
@click.option("--start_idx", "-s", type=int, default=3)
def main(agent_type: str, model_name: str, num_questions: int, start_idx: int | None):
    logger.info(
        f"\n{'=' * 100}\n"
        "Starting MusiQue Evaluation\n"
        f"Agent Type: {agent_type}\n"
        f"Model: {model_name}\n"
        f"Questions: {num_questions}\n"
        f"Start Index: {start_idx}\n"
        f"Output directory: {WORKING_DIRECTORY}\n"
        f"\n{'=' * 100}\n"
    )
    

    load_dotenv()

    if model_name == "gpt-oss":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type="gpt-oss:120b",
            url="http://127.0.0.1:7861/v1",
            model_config_dict={"temperature": 0.0},
        )
    else:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=MODEL_NAMES[model_name],
            model_config_dict={"temperature": 0.0},
        )
    agent = AGENTS[agent_type](model=model)
    # agent = bind_response_format(agent, ResearchResponseShort)


    evaluator = MusiQueEvaluator()
    dataset = load_dataset("dgslibisey/MuSiQue")
    examples = list(dataset["validation"])[start_idx : start_idx + num_questions]

    results = []
    per_f1_list, per_em_list, per_acc_list = [], [], []
    outfile = WORKING_DIRECTORY / f"{agent_type}_musique_ans_only_{start_idx}_{start_idx+num_questions}_{TIMESTAMP}.json"

    try:
        for i, ex in enumerate(tqdm(examples, desc="MuSiQue Answer-only", unit="ex", leave=True)):
            idx = start_idx + i
            question = ex.get("question", "")
            # Run agent to generate prediction

            result = run_agent_with_retry(
                agent=agent,
                # input_query=question,
                input_query = f"In `answer`, return ONLY the final short answer (entity/number/date/short noun phrase). No sentences, no prefixes, no quotes, no extra text.\nQUESTION: {question}",
                response_format=MusiQueResponse,
                max_retries=5,
            )
            # pred_answer = response.get("answer", "") if isinstance(response, dict) else ""


            response = result["response"]["answer"]
            tool_trajectory = result["tool_trajectory"]
                
            eval_req = evaluator.create_request(
                qid=ex["id"],
                answer=ex["answer"],
                answer_aliases=ex.get("answer_aliases", []),
                answerable=ex.get("answerable", True),
                prediction=response,
            )
            eval_res = evaluator.evaluate(eval_req)

            em, f1, acc = eval_res.metrics["answer_em"], eval_res.metrics["answer_f1"], eval_res.metrics["answer_acc"] 
            per_em_list.append(em)
            per_f1_list.append(f1)
            per_acc_list.append(acc)

            results.append({
                "dataset_index": idx,
                "id": ex["id"],
                "problem": question,
                "answer": ex["answer"],
                "aliases": ex.get("answer_aliases", []),
                "answerable": ex.get("answerable", True),
                "prediction": response,
                "metrics": eval_res.metrics,  # {"answer_em": ..., "answer_f1": ..., "answer_acc": ...}
                "score": eval_res.score,      # same as answer_f1
                "tool_trajectory": tool_trajectory,

            })


            if (i + 1) % 20 == 0 or (i + 1) == len(examples):
                with open(outfile, "w") as f:
                    json.dump({"results": results}, f, ensure_ascii=False, indent=2)
                tqdm.write(f"Saved to {outfile}")

            agent.reset()
            time.sleep(1)

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise

    finally:
        final_em = round(sum(per_em_list) / len(per_em_list), 3) if per_em_list else 0.0
        final_f1 = round(sum(per_f1_list) / len(per_f1_list), 3) if per_f1_list else 0.0
        final_acc = round(sum(per_acc_list) / len(per_acc_list), 3) if per_acc_list else 0.0
        summary = {"final_answer_em": final_em, "final_answer_f1": final_f1, "final_answer_acc": final_acc}
        logger.info(f"Final: {json.dumps(summary, ensure_ascii=False)}")

        with open(outfile, "w") as f:
            json.dump({"results": results, "final_metrics": summary}, f, ensure_ascii=False, indent=2)
        logger.info(f"Results written to {outfile}")


if __name__ == "__main__":
    main()