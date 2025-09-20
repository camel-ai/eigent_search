# snippet_toolkit.py

from typing import List
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
import re


class RelevanceFeedbackToolkit(BaseToolkit):
    r"""A toolkit for relevance feedback in web search, allowing agents to iteratively refine queries."""

    def __init__(self, note_taking_toolkit=None):
        super().__init__()
        self.note_taking_toolkit = note_taking_toolkit

    # def extract_relevant_details(
    #         self,
    #         snapshot: str,
    #         query: str,
    #         question: str,
    #         relevant_details: str,
    #         page_url: str = ""
    # ) -> str:
    #     r"""Call this immediately after browser_visit_page to save the relevant information you extracted.
    #
    #     IMPORTANT: Your relevant_details should directly address what the question is asking for.
    #     Before calling this function, read the snapshot and extract information that specifically
    #     answers the question - don't just copy general content from the page.
    #
    #     Args:
    #         snapshot (str): The page content from browser_visit_page (for context)
    #         query (str): The search query that led you to this page
    #         question (str): The original research question you're answering
    #         relevant_details (str): The specific information from snapshot that answers the question
    #         page_url (str): The URL of this page (for citation)
    #
    #     Returns:
    #         str: Your relevant_details, confirmed as saved
    #     """
    #     return f"Relevant details recorded:\n\n{relevant_details}"
    # def extract_relevant_details(
    #         self,
    #         snapshot: str,
    #         query: str,
    #         question: str,
    #         relevant_details: str,
    #         page_url: str = ""
    # ) -> str:
    #     r"""Call this immediately after browser_visit_page to record the information you extracted.
    #
    #     IMPORTANT: Your relevant_details must directly address what the question is asking for.
    #
    #     Before extracting, identify the core requirement of the question:
    #     - What specific type of information is being requested?
    #     - Does the snapshot contain that exact type of information?
    #     - Have you extracted the precise answer, not just related context?
    #
    #     Common mistake: Extracting general context instead of the specific answer requested.
    #     Example: If asked "who invented X", extract the person's name, not "the X company" or "scientists".
    #
    #     Ensure your extraction matches both the content AND the format of what's being asked.
    #
    #     Args:
    #         snapshot (str): The page content from browser_visit_page
    #         query (str): The search query you used
    #         question (str): The original question - identify what it's specifically asking for
    #         relevant_details (str): The specific information that directly answers the question
    #         page_url (str): The page URL
    #
    #     Returns:
    #         str: Confirmation of recorded details
    #     """
    #     return f"Relevant details recorded:\n\n{relevant_details}"
    def extract_relevant_details(
            self,
            snapshot: str,
            query: str,
            question: str,
            relevant_information: str,
            page_url: str = ""
    ) -> str:
        r"""Use this tool to extract relevant information from the page that answers the question.

        When extracting, carefully read the ENTIRE snapshot including:
        - Structured sections like tables, info boxes, and labeled fields (these often contain direct answers)
        - Main article text and paragraphs
        - All sections that might contain relevant facts

        Extraction guidelines:
        - Look for explicit, direct answers first (especially in tables/info boxes with labels like "Appointed by:", "Date:", etc.)
        - If you find conflicting or multiple pieces of information, include ALL of them - don't choose one over another
        - Include all relevant details that help answer the question
        - Be thorough - don't stop at the first piece of information you find

        The question asks for specific information - make sure your relevant_information contains that specific answer.

        Args:
            snapshot (str): The complete page content from browser_visit_page
            query (str): The search query you used to find this page
            question (str): The original question - what specific information does it ask for?
            relevant_information (str): Information you extract that answers the question (include all relevant details, even if conflicting)
            page_url (str): The URL of the page

        Returns:
            str: Confirmation that your extracted information has been recorded
        """
        return f"Relevant details recorded:\n\n{relevant_information}"

    def analyze_search_progress(
            self,
            question: str,
            current_query: str,
            findings_so_far: str,
            your_analysis: str
    ) -> str:
        r"""Call this to analyse  whether you have enough information to answer the question.

        In your_analysis parameter, write your evaluation by:
        1. Comparing your findings_so_far against what the question asks
        2. Identifying any gaps or missing details
        3. Determining if you need to refine your search

        Your analysis should explain what you have vs what you still need.

        Args:
            question (str): The original research question
            current_query (str): The search query you last used
            findings_so_far (str): All relevant details you've extracted from visited pages
            your_analysis (str): Your written evaluation - do findings answer the question completely?

        Returns:
            str: Your analysis, recorded
        """
        return f"Analysis recorded:\n\n{your_analysis}"

    def propose_query_refinement(
            self,
            question: str,
            current_query: str,
            what_you_know: str,
            what_exactly_missing: str,
            refined_query: str
    ) -> str:
        r"""Call this to create a refined search query based on identified gaps.

        Use the parameters to build your refined_query:
        - Look at what_you_know to avoid searching for information you already have
        - Focus refined_query specifically on what_exactly_missing
        - Make the query more targeted than current_query to fill the precise gap

        Your refined_query should target the specific missing piece, not repeat previous searches.

        Args:
            question (str): The original research question (for context)
            current_query (str): The search query you last used
            what_you_know (str): Summary of information you've already found
            what_exactly_missing (str): The specific information gap you identified
            refined_query (str): Your NEW search query (3-7 words) targeting what_exactly_missing

        Returns:
            str: Confirmation of your refined query to use in search_google
        """
        return f"""
        REFINED QUERY
    
        Previous Query: "{current_query}"
        Already Found: {what_you_know}
        Missing: {what_exactly_missing}
        New Query: "{refined_query}"
    
        Use this refined query in your next search_google call.
        """

    def get_tools(self) -> List[FunctionTool]:
        return [
            FunctionTool(self.extract_relevant_details),
            FunctionTool(self.analyze_search_progress),
            FunctionTool(self.propose_query_refinement),
        ]
