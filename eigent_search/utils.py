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

from camel.agents.chat_agent import ChatAgent
from camel.logger import get_logger
import tenacity

logger = get_logger(__name__)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Attempt {retry_state.outcome.attempt_number} failed: {retry_state.outcome.exception()}. Retrying in {retry_state.outcome.value} seconds..."
    ),
)
def run_agent_with_retry(
    agent: ChatAgent, problem: str, agent_type: str, hash_id: str, max_retries: int = 5
) -> dict:
    """Run agent.step with exponential retry logic.

    Args:
        agent: The agent to run
        problem: The problem text
        agent_type: Type of agent for logging
        hash_id: Hash ID for logging
        max_retries: Maximum number of retry attempts (default 5)

    Returns:
        Parsed response dict from agent

    Raises:
        Exception: If all retries fail
    """
    response = agent.step(problem)
    return eval(response.msgs[0].content)
