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

import networkx as nx
import logging
from typing import Any

from camel.toolkits import FunctionTool, BaseToolkit, SearchToolkit, ThinkingToolkit

logger = logging.getLogger(__name__)


class QueryProcessingToolkit(BaseToolkit):
    """Toolkit for processing queries for deep research.

    This toolkit provides methods for query rewriting, expansion, selection, and generation.
    Each method follows a pattern where it takes the original input and the intended output
    as parameters, then returns the intended output. This allows LLMs to understand the
    expected behavior through the docstring and call the function appropriately.
    """

    def __init__(self, initial_query: str) -> None:
        super().__init__()
        self.initial_query = initial_query
        self.frontier = {initial_query}  # set of queries to be explored
        self.explored = set()  # set of queries that have been explored
        self.trace_graph = QueryProcessingGraph(initial_query)
        self.search_tool = SearchToolkit().search_google
        self.search_counter = 0  # For counting the number of searches

    def _pop_frontier(self, query: str) -> str:
        """Pop a query from the frontier.

        Args:
            query (str): The query to pop from the frontier.

        Returns:
            str: The `query` if it has been popped from the frontier, otherwise the error message.

        NOTE: This is not a tool, but a sanity check to ensure the query is in the frontier.
        """
        try:
            self.frontier.remove(query)
        except Exception as e:
            error_message = (
                f"❌ Invalid operation: Candidate query '{query}' must be selected from the frontier.\n"
                f"📋 Current frontier:\n  - "
                + "\n  - ".join(map(str, self.frontier))
                + "\n"
                f"💡 Note: Ensure that the query is present in the frontier before removal.\n"
                f"🛑 Underlying exception: {type(e).__name__}: {e}"
            )
            logger.error(error_message)
            return error_message
        return query

    def _check_explored(self, query: str) -> str:
        """Check if the query has been explored.

        Args:
            query (str): The query to check if it has been explored.

        Returns:
            str: The `query` if it has not been explored, otherwise the error message.

        NOTE: This is not a tool, but a sanity check to ensure the query is not in the explored set.
        """
        if query in self.explored:
            error_message = (
                f"❌ Invalid operation: Query '{query}' has already been explored."
            )
            logger.error(error_message)
            return error_message
        return query

    def rewrite_query(self, query: str, rewritten_query: str) -> list[str] | str:
        """Rewrite the input query to improve search effectiveness.

        This method takes the input query and the intended rewritten version,
        then returns the rewritten query. This step is OPTIONAL and should only
        be used when the input query is vague, fuzzy, or doesn't clearly
        convey the user's intent. For clear, specific queries, this step can
        be skipped. The agent should provide a more specific, detailed, or
        differently phrased version of the original query when rewriting is needed.

        Args:
            query (str): The input query from the frontier that may need rewriting if vague or unclear.
            rewritten_query (str): The intended rewritten version of the query that should be more specific, detailed, or better phrased for search. If the input query is already clear, this can be the same as the input query.

        Returns:
            list[str] | str: The current frontier after the rewriting process or the error message.
        """

        # Pop the query from the frontier
        popped_query = self._pop_frontier(query)
        if popped_query != query:
            return popped_query  # This will be the error message from the _pop_frontier method

        # Check if the rewritten query has been explored
        checked_query = self._check_explored(rewritten_query)
        if checked_query != rewritten_query:
            return checked_query  # This will be the error message from the _check_explored method
        self.frontier.add(rewritten_query)

        # Record the process and add the popped query to explored set
        self.trace_graph.record_process(query, rewritten_query, "rewrite_query")
        self.explored.add(query)

        return list(self.frontier)

    def expand_query(self, query: str, candidate_queries: list[str]) -> list[str] | str:
        """Expand the input query into multiple related search queries.

        Expanding the query is helpful when the input query is vague, fuzzy, or doesn't clearly convey the user's intent.

        This method takes the input query and a list of intended expanded queries,
        then returns the expanded queries. The agent should expand queries using two
        main strategies:

        (1) associations - related concepts, synonyms, broader/narrower terms, or
        (2) decomposition - breaking down complex queries into simpler, more focused components.

        This helps capture different aspects and variations of the input query for comprehensive research.

        Args:
            query (str): The input query from the frontier that may need expanding.
            candidate_queries (list[str]): A list of intended expanded queries (including the original input query) that cover different aspects and variations of the input query using associations or decomposition strategies.

        Returns:
            list[str] | str: The current frontier after the expanding process or the error message.
        """
        # Pop the query from the frontier
        popped_query = self._pop_frontier(query)
        if popped_query != query:
            return popped_query  # This will be the error message from the _pop_frontier method

        # Check if the candidate queries have been explored
        checked_queries = []
        for candidate_query in candidate_queries:
            checked_query = self._check_explored(candidate_query)
            if checked_query == candidate_query:
                checked_queries.append(checked_query)
        if not checked_queries:
            return "❌ Invalid operation: All candidate queries have been explored."

        # Add the checked queries to the frontier
        self.frontier.update(checked_queries)

        # Record the process and add the popped query to explored set
        for checked_query in checked_queries:
            self.trace_graph.record_process(query, checked_query, "expand_query")
        self.explored.add(query)

        return list(self.frontier)

    def select_query_and_search(
        self, query: str, final_query: str
    ) -> list[dict[str, Any]] | str:
        """Select the best query from the frontier and perform web search.

        This method selects the most promising query from the frontier for web search. The agent should choose based on specificity, clarity, and search potential, in order to minimize the number of searches and the cost of the search. Then, optionally add advanced search operators (AND, OR, NOT, quotes, site:, filetype:, etc.) to improve search precision and relevance. Finally, perform an actual web search using the final query and return the results. The agent should return the search results as a list of strings.

        Args:
            query (str): The input query from the frontier that is selected for web search.
            final_query (str): The final query with optional advanced search operators added to the selected query that will be used for searching the web.

        Returns:
            list[dict[str, Any]] | str: The search results from the web search or the error message.
        """

        # Pop the selected query from the frontier
        popped_query = self._pop_frontier(query)
        if popped_query != query:
            return popped_query  # This will be the error message from the _pop_frontier method

        # Check if the final query has been explored
        checked_query = self._check_explored(final_query)
        if checked_query != final_query:
            return checked_query  # This will be the error message from the _check_explored method

        # NOTE: prevent searching huggingface website to avoid answer leakage
        final_query += " -site:huggingface.co"
        # Search the web
        search_results = self.search_tool(final_query)
        self.search_counter += 1

        # Record the process and add the popped query to explored set
        self.trace_graph.record_process(query, final_query, "enhance_search_query")
        self.trace_graph.record_process(final_query, search_results, "search_web")
        self.explored.add(query)
        self.explored.add(final_query)

        # NOTE: whether to remind the current frontier here? We currently think this step needs to focus on reflecting the search results, not another query from the frontier.
        return search_results

    def complete_task(
        self,
        search_results: list[dict[str, Any]],
        final_answer: str | None,
        new_queries: list[str] | None,
    ) -> tuple[str, list[dict[str, Any]]] | list[str]:
        """Complete the task if the search results are sufficient to answer the initial query.

        This method takes the search results, then returns the final answer if the task is completed, otherwise generate new queries (if applicable) to continue the research.

        Args:
            search_results (list[dict[str, Any]]): The search results from the web search.
            final_answer (str | None): The final answer if the task is completed, otherwise `None`.
            new_queries (list[str] | None): The new queries to continue the research if the task is not completed, otherwise `None`.

        Returns:
            tuple[str, list[dict[str, Any]]] | list[str]: If the task is completed, returns the final answer along with the search results that supported it; otherwise, returns the updated frontier including any newly added queries.
        """
        if final_answer:
            self.trace_graph.record_process(
                search_results, final_answer, "complete_task"
            )
            return final_answer, search_results
        elif new_queries:
            for new_query in new_queries:
                if new_query not in self.explored:
                    self.frontier.add(new_query)
                    self.trace_graph.record_process(search_results, new_query, "reflect_on_search_results")
            return list(self.frontier)
        else:
            return "❌ Invalid operation: The task is not completed. Please continue the research with the new queries."

    def get_tools(self) -> list[FunctionTool]:
        """Get the tools for the query processing toolkit.

        Note: These tools will not track usage. Use `get_query_specific_tools()`
        to get tools that track usage in a specific query graph.
        """
        return [
            FunctionTool(self.rewrite_query),
            FunctionTool(self.expand_query),
            FunctionTool(self.select_query_and_search),
            FunctionTool(self.complete_task),
            FunctionTool(ThinkingToolkit().think),
            FunctionTool(ThinkingToolkit().reflect),
        ]


class QueryProcessingGraph:
    """A graph representing the query processing process.

    The `initial_query` is the root node of the graph. The agent will call the `QueryProcessingToolkit` to process the query and return the processed query or queries. This process will be recorded as a node in the graph.
    """

    def __init__(self, initial_query: str):
        self.initial_query = initial_query
        self._graph = nx.DiGraph()  # node_id to query mapping
        self._data_to_node = {}  # map query strings to their latest node IDs
        self.new_node_id = 0
        # Add the initial query as the root node
        # NOTE: the initial query is the root node of the graph
        self._graph.add_node(self.new_node_id, data=initial_query)
        self._data_to_node[initial_query] = self.new_node_id
        self.new_node_id += 1

    @property
    def trace_graph(self) -> nx.DiGraph:
        """Get the trace graph."""
        return self._graph

    def record_process(self, from_data: str, to_data: str, action: str) -> str:
        """Record a process in the trace graph.

        Args:
            from_data (str): The source data.
            to_data (str): The target data after processing.
            action (str): The type of processing action performed.

        Returns:
            str: The node ID of the newly created target node.
        """
        # O(1) lookup for source node ID using the mapping
        source_node_id = self._data_to_node.get(from_data)

        # If source not found, create it (shouldn't happen in normal usage)
        if source_node_id is None:
            source_node_id = self.new_node_id
            self._graph.add_node(source_node_id, data=from_data)
            self._data_to_node[from_data] = source_node_id
            self.new_node_id += 1

        # Add the target node
        target_node_id = self.new_node_id
        self._graph.add_node(target_node_id, data=to_data)
        self._data_to_node[to_data] = target_node_id
        self.new_node_id += 1

        # Add the edge
        self._graph.add_edge(source_node_id, target_node_id, action=action)

        # Log the process
        logger.info(f"[{action}]: '{from_data}' -> '{to_data}'")

        return str(target_node_id)

    def render_trace_graph(self) -> str:
        """Render the process graph as a string representation."""
        if not self._graph.nodes():
            return "Empty graph"

        def format_data(data):
            """Format data for display, truncating long strings and handling complex objects."""
            if isinstance(data, str):
                # Truncate long strings
                if len(data) > 50:
                    return f'"{data[:47]}..."'
                return f'"{data}"'
            elif isinstance(data, list):
                if len(data) > 3:
                    return f"[{len(data)} items: {format_data(data[0])}, {format_data(data[1])}, ...]"
                return f"[{', '.join(format_data(item) for item in data)}]"
            elif isinstance(data, dict):
                return f"{{dict with {len(data)} keys}}"
            else:
                return str(data)

        def find_roots(graph):
            """Find all root nodes (nodes with no predecessors)."""
            return [n for n in graph.nodes() if not list(graph.predecessors(n))]

        def render_component(graph, root, visited, indent="  "):
            """Render a single connected component starting from root."""
            if root in visited:
                return []

            visited.add(root)
            node_data = graph.nodes[root].get("data", "")
            formatted_data = format_data(node_data)

            # Use human-readable node ID for display
            readable_id = str(root)
            lines = [f"{indent}{readable_id}: {formatted_data}\n"]

            # Get all successors and sort them for consistent output
            successors = sorted(graph.successors(root))
            for i, successor in enumerate(successors):
                edge_data = graph.edges[root, successor].get("action", "")
                is_last = i == len(successors) - 1

                # Add edge indicator
                prefix = "└─" if is_last else "├─"
                lines.append(f"{indent}{prefix}[{edge_data}]─>\n")

                # Recursively render successor
                if successor not in visited:
                    child_indent = indent + ("    " if is_last else "│   ")
                    lines.extend(
                        render_component(graph, successor, visited, child_indent)
                    )
                else:
                    # Handle cycles
                    lines.append(
                        f"{indent}{'    ' if is_last else '│   '}... (cycle detected)\n"
                    )

            return lines

        # Find all root nodes
        roots = find_roots(self._graph)

        if not roots:
            # If no roots found, start from any node (for cyclic graphs)
            roots = [list(self._graph.nodes())[0]]

        # Render all components
        visited = set()
        result_lines = []

        for i, root in enumerate(roots):
            if i > 0:
                result_lines.append("\n")  # Separate components
            result_lines.extend(render_component(self._graph, root, visited))

        return "".join(result_lines)
