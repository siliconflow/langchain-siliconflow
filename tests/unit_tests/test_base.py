from langchain_siliconflow.chat_models import ChatSiliconFlow
import pytest
from typing import Any, Literal, Optional, Union, cast
from pydantic import BaseModel, Field, SecretStr

class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


class MakeASandwich(BaseModel):
    "Make a sandwich given a list of ingredients."

    bread_type: str
    cheese_type: str
    condiments: list[str]
    vegetables: list[str]


@pytest.mark.parametrize(
    "tool_choice",
    [
        "any",
        "none",
        "auto",
        "required",
        "GenerateUsername",
        {"type": "function", "function": {"name": "MakeASandwich"}},
        False,
        None,
    ],
)
@pytest.mark.parametrize("strict", [True, False, None])
def test_bind_tools_tool_choice(tool_choice: Any, strict: Optional[bool]) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3", temperature=0)
    llm.bind_tools(
        tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice, strict=strict
    )
