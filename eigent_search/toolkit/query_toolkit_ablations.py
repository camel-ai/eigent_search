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
"""Ablation variants of QueryProcessingToolkit for experiments."""

from functools import wraps

from camel.toolkits.function_tool import FunctionTool

from .query_toolkit import QueryProcessingToolkit


def with_custom_docstring(new_docstring: str):
    """Decorator to wrap a method with a custom docstring while preserving behavior."""

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            return method(self, *args, **kwargs)

        wrapper.__doc__ = new_docstring
        return wrapper

    return decorator


class Fixed10ResultsToolkit(QueryProcessingToolkit):
    """Ablation: Only select_query_and_search with fixed 10 results.

    This ablation removes all query refinement, expansion, and analysis tools,
    leaving only the basic search functionality with fixed 10 results per search.
    """

    @with_custom_docstring(
        r"""Create a query based on the original query and current research progress, and perform web search.
        The agent should generate query based on specificity, clarity, and search potential,
        in order to minimize the number of searches and the cost of the search.
        If the search and the corresponding web browsing results are not sufficient to answer the user's initial query,
        the agent should create another query and perform web search again.

        Args:
            query (str): The search query to use for web search. Choose your query
                carefully to maximize the chance of finding relevant results.
        Returns:
            dict[str, dict[str, str]]: The search results from the web search. The key is "search_results" and the value is a dict where each key is a URL and each value is the string of the title, description, and long description of the result. If the search fails, the key is "None" and the value is the error message.
        """
    )
    def select_query_and_search(self, query: str) -> dict[str, dict[str, str]]:
        return super().select_query_and_search(query=query)

    def get_tools(self) -> list[FunctionTool]:
        """Returns only the select_query_and_search tool."""
        return [FunctionTool(self.select_query_and_search)]


class NoQueryToolsToolkit(QueryProcessingToolkit):
    """Ablation: Search + analyze + extract, but no query refinement tools.

    This ablation removes the query refinement and expansion tools
    (local_expand_query, local_refine_query, global_refine_query, global_expand_query)
    while keeping the search, analysis, and extraction capabilities.
    """

    @with_custom_docstring(
        r"""Create a query based on the original query and current research progress, and perform web search.
        The agent should generate query based on specificity, clarity, and search potential,
        in order to minimize the number of searches and the cost of the search.
        If the search and the corresponding web browsing results are not sufficient to answer the user's initial query,
        the agent should create another query and perform web search again.

        Args:
            query (str): The search query to use for web search. Choose your query
                carefully to maximize the chance of finding relevant results.
        Returns:
            dict[str, dict[str, str]]: The search results from the web search. The key is "search_results" and the value is a dict where each key is a URL and each value is the string of the title, description, and long description of the result. If the search fails, the key is "None" and the value is the error message.
        """
    )
    def select_query_and_search(self, query: str) -> dict[str, dict[str, str]]:
        return super().select_query_and_search(query=query)

    def get_tools(self) -> list[FunctionTool]:
        """Returns search, analysis, and extraction tools without query tools."""
        return [
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.analyze_search_progress),
            FunctionTool(self.extract_relevant_details),
        ]


class QueryToolsOnlyToolkit(QueryProcessingToolkit):
    """Ablation: Search + query tools, but no analyze/extract.

    This ablation removes the analysis and extraction tools
    (analyze_search_progress, extract_relevant_details) while keeping
    the search and all query refinement/expansion capabilities.
    """

    def get_tools(self) -> list[FunctionTool]:
        """Returns search and query tools without analyze/extract."""
        return [
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.local_expand_query),
            FunctionTool(self.local_refine_query),
            FunctionTool(self.global_refine_query),
            FunctionTool(self.global_expand_query),
        ]
