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
import logging

from camel.toolkits import FunctionTool, BaseToolkit, SearchToolkit, ThinkingToolkit

logger = logging.getLogger(__name__)


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
        self.search_tool = SearchToolkit().search_google
        self.search_counter = 0  # For counting the number of searches
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
            query (str): The input query that may need rewriting if vague or unclear.
            rewritten_query (str): The intended rewritten version of the query that should be more specific, detailed, or better phrased for search. If the input query is already clear, this can be the same as the input query.

        Returns:
            str: The rewritten query as provided by the agent.
        """
        from_id = self._add_process_node(query)
        to_id = self._add_process_node(rewritten_query)
        self._add_transformation_edge(from_id, to_id, "rewrite")
        logger.info(f"[rewrite]: '{query}' -> '{rewritten_query}'")
        return rewritten_query

    def expand_query(self, query: str, candidate_queries: list[str]) -> list[str]:
        """Expand the input query into multiple related search queries.
        
        Expanding the query is helpful when the input query is vague, fuzzy, or doesn't clearly convey the user's intent. 

        This method takes the input query and a list of intended expanded queries,
        then returns the expanded queries. The agent should expand queries using two
        main strategies:

        (1) associations - related concepts, synonyms, broader/narrower terms, or
        (2) decomposition - breaking down complex queries into simpler, more focused components.

        This helps capture different aspects and variations of the input query for comprehensive research.

        Args:
            query (str): The input query to be expanded.
            candidate_queries (list[str]): A list of intended expanded queries (including the original input query) that cover different aspects and variations of the input query using associations or decomposition strategies.

        Returns:
            list[str]: The list of expanded queries as provided by the agent.
        """
        # Expand the query
        candidate_queries = (
            [query, *candidate_queries]
            if query not in candidate_queries
            else candidate_queries
        )
        # Track the expanded queries list as a single node
        from_id = self._add_process_node(query)
        to_id = self._add_process_node(candidate_queries)
        self._add_transformation_edge(from_id, to_id, "expand")
        logger.info(f"[expand]: '{query}' -> '{candidate_queries}'")
        return candidate_queries

    def select_query_and_search(
        self, candidate_queries: list[str], selected_query: str, final_query: str
    ) -> list[dict[str, Any]]:
        """Select the best query from candidates and perform web search.

        This method takes candidate queries and selects the most promising one for web search. The agent should choose based on specificity, clarity, and search potential. Then, optionally add advanced search operators (AND, OR, NOT, quotes, site:, filetype:, etc.) to improve search precision and relevance. Finally, perform an actual web search using the final query and return the results. The agent should return the search results as a list of strings.

        Args:
            candidate_queries (list[str]): The list of candidate queries to choose from.
            selected_query (str): The query selected from the candidate queries.
            final_query (str): The final query with optional advanced search operators added to the selected query that will be used for searching the web. 

        Returns:
            list[dict[str, Any]]: The search results from the web search.
        """
        # Select the best query from candidates
        from_id = self._add_process_node(candidate_queries)
        to_id = self._add_process_node(selected_query)
        self._add_transformation_edge(from_id, to_id, "select")
        logger.info(f"[select]: '{candidate_queries}' -> '{selected_query}'")
        # Refine the selected query
        from_id = self._add_process_node(selected_query)
        to_id = self._add_process_node(final_query)
        self._add_transformation_edge(from_id, to_id, "refine")
        logger.info(f"[refine]: '{selected_query}' -> '{final_query}'")
        # Search the web
        # NOTE: ad-hoc fix to prevent searching huggingface website
        final_query += " -site:huggingface.co"
        search_results = self.search_tool(final_query)
        self.search_counter += 1
        from_id = self._add_process_node(final_query)
        to_id = self._add_process_node(search_results)
        self._add_transformation_edge(from_id, to_id, "search")
        logger.info(f"[search]: '{final_query}' -> '{search_results}'")
        return search_results

    def generate_new_queries(
        self, search_results: list[dict[str, Any]], new_queries: list[str]
    ) -> list[str]:
        """Generate new queries based on the search results.

        This method takes search results and a list of intended new queries,
        then returns the new queries. The agent should analyze the search results
        and provide follow-up queries that would help gather more specific information.

        Args:
            search_results (list[dict[str, Any]]): The results from previous web searches.
            new_queries (list[str]): A list of intended new queries generated based on the search results that would help gather more specific or related information.

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
            FunctionTool(ThinkingToolkit().think),
            FunctionTool(ThinkingToolkit().reflect),
            # FunctionTool(self.generate_new_queries),
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

    def render_process_graph(self) -> str:
        """Render the process graph as a string representation."""
        if not self.process_graph.nodes():
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

            lines = [f"{indent}{root}: {formatted_data}\n"]

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
        roots = find_roots(self.process_graph)

        if not roots:
            # If no roots found, start from any node (for cyclic graphs)
            roots = [list(self.process_graph.nodes())[0]]

        # Render all components
        visited = set()
        result_lines = []

        for i, root in enumerate(roots):
            if i > 0:
                result_lines.append("\n")  # Separate components
            result_lines.extend(render_component(self.process_graph, root, visited))

        return "".join(result_lines)
