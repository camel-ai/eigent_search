# dev/boerz/examples/run_agent.py
"""
MVP Example: Simple demonstration of Boerz's research agent.

This example shows how to create and use the research agent with a basic query.
"""

import sys
import os
import logging

# Add the project root to the path so we can import from dev.boerz
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Configure logging to display to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set all loggers to WARNING level (mute INFO and DEBUG)
logging.getLogger().setLevel(logging.WARNING)

# Only show INFO logs from our researcher_instance module
logging.getLogger('dev.boerz.researcher_instance').setLevel(logging.INFO)

from dev.boerz.researcher_instance import LeadResearcher, JuniorResearcher
from librarian.research.research_toolkit import ResearchToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


def main():
    """Run a simple research example."""
    print("🤖 Boerz's Research Agent MVP Demo")
    print("=" * 50)
    
    # Create a simple research toolkit
    toolkit = ResearchToolkit()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.5},
    )
    # Create the lead researcher
    # Note: For MVP, we'll use a dummy model - in production you'd use a real LLM
    lead_researcher = LeadResearcher(
        system_message="You are a lead research agent. Plan and coordinate research tasks effectively.",
        model=model,  # Dummy model for MVP
        research_toolkit=toolkit
    )
    
    # Example research query
    query = "What is machine learning?"
    
    print(f"🔍 Research Query: {query}")
    print("⏳ Starting research...")
    
    try:
        # Perform the research
        result = lead_researcher.research(query)
        
        print("\n✅ Research Complete!")
        print("=" * 50)
        print(f"📝 Answer: {result.get('answer', 'No answer generated')}")
        print(f"📚 Evidence: {result.get('evidence', [])}")
        
    except Exception as e:
        print(f"❌ Research failed: {e}")
        print("This is expected in MVP since we're using dummy components.")


def demo_junior_researcher():
    """Demo the junior researcher directly."""
    print("\n🧪 Junior Researcher Demo")
    print("=" * 30)
    
    from dev.boerz.researcher_instance import JuniorResearcher
    
    toolkit = ResearchToolkit()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1_MINI,
        model_config_dict={"temperature": 0.5},
    )
    junior = JuniorResearcher(
        system_message="You are a focused junior researcher.",
        model=model,
        research_toolkit=toolkit
    )
    
    query = "What is Python programming?"
    result = junior.research(query)
    
    print(f"Query: {query}")
    print(f"Result: {result}")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Run the junior researcher demo
    # demo_junior_researcher()
    
