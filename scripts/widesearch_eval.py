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


from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import click
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

from camel.agents import ChatAgent
from camel.logger import get_logger, set_log_file, set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


from eigent_search.baseline import (
    ChainOfThoughtAgent,
    DirectAnswerAgent,
    KnowledgeThenReasoningAgent,
    SimpleResearchAgent,
)
from eigent_search.research import deep_search_agent_factory
from eigent_search.utils import run_agent_with_retry


from eigent_search.evaluation.widesearch.data_loader import (
    WideSearchDataLoaderHF,
    WideSearchQuery,
    WideSearchResponse,
)
from eigent_search.evaluation.widesearch.evaluation import evaluate_single_query


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WORKING_DIRECTORY = Path(
    f"eigent_search_results_{TIMESTAMP}",
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
    "gpt-oss": "gpt-oss:120b",  
}


set_log_file(WORKING_DIRECTORY / "widesearch_eval.log")
set_log_level(logging.INFO)
logger = get_logger(__name__)



# class WideSearchAgentResponse(BaseModel):
# #     answer: str = Field(..., description="The answer to the research question.")
# #     search_results: Optional[List[str]] = Field(
# #         default=None, description="The search results that lead to the answer."
# #     )
# #     # trace: Optional[Dict[str, Any]] = None
#     answer: str = Field(..., description="The answer to the research question.")
#     search_results: list[str] = Field(
#         ..., description="The search results that lead to the answer."
#     )



class Trace(BaseModel):
    # 明确定义允许的字段；不要用 dict[str, Any]
    steps: Optional[List[str]] = None
    tool_calls: Optional[List[str]] = None
    model_config = ConfigDict(extra='forbid')  # 关键：封闭

class WideSearchAgentResponse(BaseModel):
    answer: str = Field(..., description="The answer to the research question.")
    search_results: Optional[List[str]] = Field(
        default=None, description="The search results that lead to the answer."
    )
    trace: Optional[Trace] = None   # 用封闭的 Trace，而不是 dict[Any]
    model_config = ConfigDict(extra='forbid')  # 顶层也封闭



@click.command()
@click.option("--agent_type", "-a", type=click.Choice(AGENTS.keys()), required=True)
@click.option(
    "--model_name", "-m", type=click.Choice(MODEL_NAMES.keys()), default="gpt-4.1-mini"
)
@click.option("--num_questions", "-n", type=int, default=5)
@click.option("--start_idx", "-s", type=int, default=0, help="Start index for the test samples.")
@click.option(
    "--trial_num",
    "-t",
    type=int,
    default=1,
    help="the number of replicates for each example（aligned with best-of/avg-of n）",
)



def main(agent_type: str, model_name: str, num_questions: int, start_idx: int, trial_num: int):
    # Log evaluation configuration
    logger.info(
        "\n" + "=" * 100 + "\n"
        f"Starting WideSearch Evaluation\n"
        f"Agent: {agent_type}\n"
        f"Model: {model_name}\n"
        f"Questions: {num_questions}\n"
        f"Start Index: {start_idx}\n"
        f"Trials per example: {trial_num}\n"
        f"Output directory: {WORKING_DIRECTORY}\n"
        + "=" * 100 + "\n"
    )

    load_dotenv()
    if model_name == "gpt-oss":
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type="gpt-oss:120b",
            url="http://129.212.188.6:7861/v1",
            model_config_dict={"temperature": 0.0},
        )
    else:
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=MODEL_NAMES[model_name],
            model_config_dict={"temperature": 0.0},
        )
    
    agent = AGENTS[agent_type](model=model)

    # eval_model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OPENAI,
    #     model_type=ModelType.GPT_4_1_MINI,
    #     model_config_dict={"temperature": 0.0},
    # )
    # eval_agent = ChatAgent(model=eval_model)

    # load the dataset
    data_loader = WideSearchDataLoaderHF()
    all_instance_ids = data_loader.get_instance_id_list()

    test_samples_ids: List[str]
    test_samples_ids = all_instance_ids[start_idx : start_idx + num_questions]

    if not test_samples_ids:
        logger.warning("No instances selected. Check --start_idx/--num_questions")
        return

    # output file name, including the search results and score summary for each example
    exact_name = f"agent={agent_type}_model={model_name}_n={len(test_samples_ids)}_trial={trial_num}"
    per_run_json = WORKING_DIRECTORY / f"widesearch_eval_{exact_name}.json"
    summary_json = WORKING_DIRECTORY / f"widesearch_eval_{exact_name}_summary.json"
    results: List[Dict[str, Any]] = []

    tool_trajectory_dir = WORKING_DIRECTORY / "tool_trajectories"
    tool_trajectory_dir.mkdir(exist_ok=True, parents=True)
    

    total_token_usage = 0
    try:
        for idx, instance_id in enumerate(
            tqdm(test_samples_ids, desc="WideSearch Evaluation", unit="instance", leave=True)
            ):
            query: WideSearchQuery = data_loader.load_query_by_instance_id(instance_id)

            # each example will conduct muti trials
            trial_metrics = []
            trial_csv_paths = []

            for trial_idx in range(trial_num):
                run = run_agent_with_retry(
                    agent=agent,
                    input_query=query.query,  # WideSearchQuery.query is plain text of the question.
                    response_format=WideSearchAgentResponse,
                    # response_format=None,
                    max_retries=2,
                    timeout_minutes=5,
                )
                resp: Dict[str, Any] = run["response"]
                token_usage: int = run.get("token_usage", 0)
                total_token_usage += token_usage
                logger.info("Total token usage so far: %d", total_token_usage)

                # extract the answer text
                if isinstance(resp, dict) and "answer" in resp:
                    pred_text = resp["answer"]
                else:
                    # if failed, convert the json to text directly
                    pred_text = json.dumps(resp, ensure_ascii=False)

                # for better evaluation, transform the data format to WideSearchResponse style
                ws_response = WideSearchResponse(
                    instance_id=instance_id,
                    response=pred_text,
                    messages=[],   # you could also pass the history messages of agent
                    trial_idx=trial_idx,
                )

                # csv output path
                trial_csv = WORKING_DIRECTORY / f"{model_name}_{instance_id}_{trial_idx}_eval_result.csv"
                eval_result = evaluate_single_query(
                    query, ws_response, str(trial_csv), eval_model_config_name="default_eval_config"
                )
                trial_csv_paths.append(str(trial_csv))
                trial_metrics.append(eval_result)

    
            results.append(
                {
                    "instance_id": instance_id,
                    "query": query.query,
                    "language": getattr(query, "language", "unknown"),
                    "trials": [
                        {
                            "trial_idx": i,
                            "score": float(m.score),
                            "precision_by_row": m.precision_by_row,
                            "recall_by_row": m.recall_by_row,
                            "f1_by_row": m.f1_by_row,
                            "precision_by_item": m.precision_by_item,
                            "recall_by_item": m.recall_by_item,
                            "f1_by_item": m.f1_by_item,
                            "eval_csv": trial_csv_paths[i],
                        }
                        for i, m in enumerate(trial_metrics)
                    ],
                    "token_usage": token_usage,
                    "total_token_usage": total_token_usage,
                }
            )

            # 滚动输出与周期性保存
            current_precision_by_row = trial_metrics[-1].precision_by_row
            current_recall_by_row = trial_metrics[-1].recall_by_row
            current_f1_by_row = trial_metrics[-1].f1_by_row
            current_precision_by_item = trial_metrics[-1].precision_by_item
            current_recall_by_item = trial_metrics[-1].recall_by_item
            current_f1_by_item = trial_metrics[-1].f1_by_item
            
            logger.info(
                f"\nQuestion: {idx + 1} / {len(test_samples_ids)}\n"
                f"Agent: {agent_type}\n"
                f"current_precision_by_row:{current_precision_by_row}\n"
                f"current_recall_by_row:{current_recall_by_row}\n"
                f"current_f1_by_row:{current_f1_by_row}\n"
                f"current_precision_by_item:{current_precision_by_item}\n"
                f"current_recall_by_item:{current_recall_by_item}\n"
                f"current_f1_by_item:{current_f1_by_item}\n"

            )

            # Save tool trajectory of every trials for this problem
            trajectory_file = (
                tool_trajectory_dir / f"problem_{instance_id}_{trial_idx}_trajectory.json"
            )
            with open(trajectory_file, "w") as f:
                json.dump(run.get("tool_trajectory", []), f, indent=2)
            logger.info(f"Tool trajectory saved to {trajectory_file} ...")

            #  Save results periodically
            if (idx + 1) % 2 == 0 or (idx + 1) == len(test_samples_ids):
                with open(per_run_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Results saved to {per_run_json}")

            agent.reset()

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # aggregate the trials within each example: calculate avg / max / min, and then take the average of all examples
        metrics_keys = [
            "score",
            "precision_by_row",
            "recall_by_row",
            "f1_by_row",
            "precision_by_item",
            "recall_by_item",
            "f1_by_item",
        ]


        per_instance, summary = aggregate_trials_then_instances(
            results=results,
            metrics_keys=metrics_keys,
        )

        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary saved to {summary_json}")
        logger.info(f"Token usage total: {total_token_usage}")

def aggregate_trials_then_instances(
    results: List[Dict[str, Any]],
    metrics_keys: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    先对每个问题(instance)聚合其 trials 的各项指标(avg/min/max)，
    再对所有问题做宏平均(按问题数平均，而不是按trial数加权)。

    Returns:
        per_instance: List[ {instance_id, <metric>: {avg_n,max_n,min_n}, ... } ]
        summary: { metric: {avg_n, max_n, min_n}, ... }  # 宏平均（按问题）
    """
    per_instance: List[Dict[str, Any]] = []
    # 收集每个问题的统计，用于最后做“按问题宏平均”
    per_metric_buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k in metrics_keys}

    for rec in results:
        trials = rec.get("trials", []) or []
        # 收集该问题每个metric在不同trial上的值
        vals_by_metric: Dict[str, List[float]] = {k: [] for k in metrics_keys}

        for t in trials:
            # 指标都在trial顶层
            for k in metrics_keys:
                v = t.get(k, None)
                if v is None:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    # 非数值/无法转换的跳过
                    continue
                if math.isnan(v):
                    continue
                vals_by_metric[k].append(v)

        inst_stats: Dict[str, Any] = {
            "instance_id": rec.get("instance_id"),
        }

        for k in metrics_keys:
            vals = vals_by_metric[k]
        
            avg_v = sum(vals) / len(vals)
            max_v = max(vals)
            min_v = min(vals)
            stat = {"avg_n": avg_v, "max_n": max_v, "min_n": min_v}
            inst_stats[k] = stat
            per_metric_buckets[k].append(stat)
        per_instance.append(inst_stats)

    # 按问题宏平均（对每个问题的 avg/max/min 再做平均）
    summary: Dict[str, Dict[str, float]] = {}
    for k in metrics_keys:
        bucket = per_metric_buckets[k]
        if not bucket:
            continue
        summary[k] = {
            "avg_n": sum(x["avg_n"] for x in bucket) / len(bucket),
            "max_n": sum(x["max_n"] for x in bucket) / len(bucket),
            "min_n": sum(x["min_n"] for x in bucket) / len(bucket),
        }

    return per_instance, summary

if __name__ == "__main__":
    main()