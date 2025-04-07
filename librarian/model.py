from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv


def create_openai_model(
    model_type: ModelType = ModelType.GPT_4O_MINI,
    model_config_dict: dict = {"temperature": 0.5},
):
    """Create an OpenAI model."""
    load_dotenv()  # load the openai key from .env
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_type,
        model_config_dict=model_config_dict,
    )
