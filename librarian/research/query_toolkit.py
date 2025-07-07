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

from typing import Any
import networkx as nx

from camel.toolkits import FunctionTool, BaseToolkit, SearchToolkit


class QueryProcessingToolkit(BaseToolkit):
    """Toolkit for processing queries for deep research.

    This toolkit provides methods for query rewriting, expansion, selection, and generation.
    Each method follows a pattern where it takes the original input and the intended output
    as parameters, then returns the intended output. This allows LLMs to understand the
    expected behavior through the docstring and call the function appropriately.
    """

    def __init__(self) -> None:
        super().__init__()
        self.process_graph = nx.DiGraph()
        self.search_tool = FunctionTool(SearchToolkit().search_google)
        self.process_counter = 0  # For generating unique IDs

    def rewrite_query(self, query: str, rewritten_query: str) -> str:
        """Rewrite the input query to improve search effectiveness.

        This method takes the input query and the intended rewritten version,
        then returns the rewritten query. This step is OPTIONAL and should only
        be used when the input query is vague, fuzzy, or doesn't clearly
        convey the user's intent. For clear, specific queries, this step can
        be skipped. The agent should provide a more specific, detailed, or
        differently phrased version of the original query when rewriting is needed.

        Args:
            query: The input query that may need rewriting if vague or unclear.
            rewritten_query: The intended rewritten version of the query that should be more specific, detailed, or better phrased for search. If the input query is already clear, this can be the same as the input query.

        Returns:
            str: The rewritten query as provided by the agent.
        """
        from_id = self._add_process_node(query)
        to_id = self._add_process_node(rewritten_query)
        self._add_transformation_edge(from_id, to_id, "rewrite")
        return rewritten_query

    def expand_query(self, query: str, expanded_queries: list[str]) -> list[str]:
        """Expand the input query into multiple related search queries.

        This method takes the input query and a list of intended expanded queries,
        then returns the expanded queries. The agent should expand queries using two
        main strategies:

        (1) associations - related concepts, synonyms, broader/narrower terms, or
        (2) decomposition - breaking down complex queries into simpler, more focused components.

        This helps capture different aspects and variations of the input query for comprehensive research.

        Args:
            query: The input query to be expanded.
            expanded_queries: A list of intended expanded queries (including the original input query) that cover different aspects and variations of the input query using associations or decomposition strategies.

        Returns:
            list[str]: The list of expanded queries as provided by the agent.
        """
        # Expand the query
        result_queries = (
            [query, *expanded_queries]
            if query not in expanded_queries
            else expanded_queries
        )
        # Track the expanded queries list as a single node
        from_id = self._add_process_node(query)
        to_id = self._add_process_node(result_queries)
        self._add_transformation_edge(from_id, to_id, "expand")

        return result_queries

    def select_query_and_search(
        self, candidate_queries: list[str], selected_query: str, final_query: str
    ) -> list[dict[str, Any]]:
        """Select the best query from candidates and perform web search.

        This method takes candidate queries and selects the most promising one for web search. The agent should choose based on specificity, clarity, and search potential. Then, optionally add advanced search operators (quotes, site:, filetype:, etc.) to improve search precision and relevance. Finally, perform an actual web search using the final query and return the results. The agent should return the search results as a list of strings.

        Args:
            candidate_queries: The list of candidate queries to choose from.
            selected_query: The query selected from the candidate queries.
            final_query: The final query with optional advanced search operators added to the selected query that will be used for searching the web.

        Returns:
            list[dict[str, Any]]: The search results from the web search.
        """
        # Select the best query from candidates
        from_id = self._add_process_node(candidate_queries)
        to_id = self._add_process_node(selected_query)
        self._add_transformation_edge(from_id, to_id, "select")
        # Refine the selected query
        from_id = self._add_process_node(selected_query)
        to_id = self._add_process_node(final_query)
        self._add_transformation_edge(from_id, to_id, "refine")
        # Search the web
        search_results = self.search_tool(final_query)
        from_id = self._add_process_node(final_query)
        to_id = self._add_process_node(search_results)
        self._add_transformation_edge(from_id, to_id, "search")
        return search_results

    def generate_new_queries(
        self, search_results: list[Any], new_queries: list[str]
    ) -> list[str]:
        """Generate new queries based on the search results.

        This method takes search results and a list of intended new queries,
        then returns the new queries. The agent should analyze the search results
        and provide follow-up queries that would help gather more specific information.

        Args:
            search_results: The results from previous web searches.
            new_queries: A list of intended new queries generated based on the search results
                        that would help gather more specific or related information.

        Returns:
            list[str]: The list of new queries as provided by the agent.
        """
        from_id = self._add_process_node(search_results)
        to_id = self._add_process_node(new_queries)
        self._add_transformation_edge(from_id, to_id, "generate")
        return new_queries

    def get_tools(self) -> list[FunctionTool]:
        """Get the tools for the query processing toolkit."""
        return [
            FunctionTool(self.rewrite_query),
            FunctionTool(self.expand_query),
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.generate_new_queries),
        ]

    def _add_process_node(self, data: Any) -> str:
        """Add a process node to the graph and return its ID."""
        process_id = f"process_{self.process_counter}"
        self.process_counter += 1

        self.process_graph.add_node(process_id, data=data)
        return process_id

    def _add_transformation_edge(self, from_id: str, to_id: str, action: str) -> None:
        """Add a transformation edge to the graph."""
        self.process_graph.add_edge(from_id, to_id, action=action)
