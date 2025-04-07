from pydantic import BaseModel, Field
from typing import Literal


class LibrarianResponse(BaseModel):
    """The response from the librarian agent."""

    knowledge: list[str] = Field(..., description="The retrieved knowledge.")
    reasoning: str = Field(..., description="The step-by-step reasoning process.")
    answer: str = Field(..., description="The final answer.")


class PlainResponse(BaseModel):
    """The response from the plain agent."""

    answer: str = Field(..., description="The final answer.")


class CoTResponse(BaseModel):
    """The response from the CoT agent."""

    reasoning: str = Field(..., description="The step-by-step reasoning process.")
    answer: str = Field(..., description="The final answer.")


# Grade constants
GRADE_CORRECT = "CORRECT"
GRADE_INCORRECT = "INCORRECT"
GRADE_NOT_ATTEMPTED = "NOT_ATTEMPTED"


class Grade(BaseModel):
    """The grade of the predicted answer for LLM-as-Judge."""

    grade: Literal[GRADE_CORRECT, GRADE_INCORRECT, GRADE_NOT_ATTEMPTED] = Field(
        ..., description="The grade of the predicted answer."
    )
