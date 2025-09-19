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

import re
from typing import List
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool


class RelevanceFeedbackToolkit(BaseToolkit):
    r"""A toolkit for relevance feedback in web search, allowing agents to iteratively refine queries."""

    def __init__(self):
        super().__init__()

    def extract_relevant_details(
            self,
            snapshot: str,
            query: str,
            question: str
    ) -> str:
        r"""Agent should use this after visiting a page to extract relevant details for relevance feedback.
        This function helps the agent provide feedback on the relevance of the visited document by
        cleaning the page content and returning text that contains details relevant to the query.
        These details will be used to refine the search query in subsequent iterations.

        Args:
            snapshot (str): The page snapshot from browser_visit_page or browser_get_page_snapshot
            query (str): Your current search query
            question (str): The original research question

        Returns:
            str: The extracted relevant details from the page, cleaned and ready for analysis
        """
        # Clean text - remove ref markers and excessive whitespace
        text = re.sub(r'\[ref=\d+\]', '', snapshot)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def local_adjust_query(
            self,
            original_query: str,
            question: str,
            relevant_details: str,
            adjusted_query: str
    ) -> str:
        r"""Agent must call this after extract_relevant_details.

        Use the output from extract_relevant_details as relevant_details.
        Agent should adjust the query based on what information gaps remain between the
        relevant_details and what's needed to answer the question.

        Args:
            original_query (str): The original/current search query
            question (str): The original question/problem
            relevant_details (str): Output from extract_relevant_details
            adjusted_query (str): Your proposed adjusted query

        Returns:
            str: The adjusted query for the next search iteration
        """
        return adjusted_query

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(self.extract_relevant_details),
            FunctionTool(self.local_adjust_query),
        ]