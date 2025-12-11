"""Test ChatOpenAI chat model."""

import base64
import json
import os
from collections.abc import AsyncIterator
from typing import Literal, Optional, cast

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_tests.integration_tests.chat_models import (
    _validate_tool_call_message,
    magic_function,
)
from pydantic import BaseModel, Field

from langchain_siliconflow.chat_models import ChatSiliconFlow as ChatOpenAI

from .callbacks import FakeCallbackHandler

MAX_TOKEN_COUNT = 16
MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"
VISION_MODEL_NAME = "zai-org/GLM-4.5V"


@pytest.mark.scheduled
def test_chat_openai() -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(
        temperature=0.7,
        base_url=os.getenv(
            "SILICONFLOW_BASE_URL", default="https://api.siliconflow.com/v1"
        ),
        organization=None,
        openai_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_openai_model() -> None:
    """Test ChatOpenAI wrapper handles model_name."""
    chat = ChatOpenAI(model="foo")
    assert chat.model_name == "foo"
    chat = ChatOpenAI(model_name="bar")  # type: ignore[call-arg]
    assert chat.model_name == "bar"


@pytest.mark.parametrize("use_responses_api", [False])
def test_chat_openai_system_message(use_responses_api: bool) -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(use_responses_api=use_responses_api, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.text(), str)


@pytest.mark.scheduled
@pytest.mark.parametrize("use_responses_api", [False])
def test_chat_openai_streaming(use_responses_api: bool) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatOpenAI(
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
        verbose=True,
        use_responses_api=use_responses_api,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, streaming=True)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


@pytest.mark.scheduled
async def test_async_chat_openai_bind_tools() -> None:
    """Test ChatOpenAI wrapper with bind_tools."""

    class Person(BaseModel):
        """Identifying information about a person."""

        name: str = Field(..., title="Name", description="The person's name")
        age: int = Field(..., title="Age", description="The person's age")
        fav_food: Optional[str] = Field(
            default=None, title="Fav Food", description="The person's favorite food"
        )

    chat = ChatOpenAI(max_tokens=30, n=1, streaming=True).bind_tools(  # type: ignore[call-arg]
        tools=[Person], tool_choice={"type": "function", "function": {"name": "Person"}}
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", "Use the provided Person function"), ("user", "{input}")]
    )

    chain = prompt | chat

    message = HumanMessage(content="Sally is 13 years old")
    response = await chain.abatch([{"input": message}])

    assert isinstance(response, list)
    assert len(response) == 1
    for generation in response:
        assert isinstance(generation, AIMessage)


@pytest.mark.scheduled
@pytest.mark.parametrize("use_responses_api", [False])
async def test_openai_abatch_tags(use_responses_api: bool) -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_tokens=MAX_TOKEN_COUNT, use_responses_api=use_responses_api)  # type: ignore[call-arg]

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.text(), str)


@pytest.mark.flaky(retries=3, delay=1)
def test_openai_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        service_tier="flex",  # Also test service_tier
        max_retries=3,  # Add retries for 503 capacity errors
    )

    result = llm.invoke("Hello", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)

    # assert no response headers if include_response_headers is not set
    assert "headers" not in result.response_metadata


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI()

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("finish_reason") is not None
    assert full.response_metadata.get("model_name") is not None

    # check token usage
    aggregate: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    for chunk in llm.stream("Hello", stream_usage=True):
        assert isinstance(chunk.content, str)
        aggregate = chunk if aggregate is None else aggregate + chunk
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata is not None:
            chunks_with_token_counts += 1
        if chunk.response_metadata:
            chunks_with_response_metadata += 1
    assert isinstance(aggregate, AIMessageChunk)
    assert aggregate.usage_metadata is not None
    assert aggregate.usage_metadata["input_tokens"] > 0
    assert aggregate.usage_metadata["output_tokens"] > 0
    assert aggregate.usage_metadata["total_tokens"] > 0


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""

    async def _test_stream(stream: AsyncIterator, expect_usage: bool) -> None:
        full: Optional[BaseMessageChunk] = None
        chunks_with_token_counts = 0
        chunks_with_response_metadata = 0
        async for chunk in stream:
            assert isinstance(chunk.content, str)
            full = chunk if full is None else full + chunk
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                chunks_with_token_counts += 1
            if chunk.response_metadata:
                chunks_with_response_metadata += 1
        assert isinstance(full, AIMessageChunk)
        assert full.response_metadata.get("finish_reason") is not None
        assert full.response_metadata.get("model_name") is not None

    llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKEN_COUNT)  # type: ignore[call-arg]
    await _test_stream(llm.astream("Hello"), expect_usage=False)
    await _test_stream(
        llm.astream("Hello", stream_options={"include_usage": True}), expect_usage=True
    )
    await _test_stream(llm.astream("Hello", stream_usage=True), expect_usage=True)
    llm = ChatOpenAI(
        temperature=0,
        max_tokens=MAX_TOKEN_COUNT,  # type: ignore[call-arg]
        model_kwargs={"stream_options": {"include_usage": True}},
    )
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    # await _test_stream(
    #     llm.astream("Hello", stream_options={"include_usage": False}),
    #     expect_usage=False,
    # )
    llm = ChatOpenAI(temperature=0, max_tokens=MAX_TOKEN_COUNT, stream_usage=True)  # type: ignore[call-arg]
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    await _test_stream(llm.astream("Hello", stream_usage=False), expect_usage=False)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


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


@pytest.mark.parametrize("use_responses_api", [False])
def test_manual_tool_call_msg(use_responses_api: bool) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(
        model=MODEL_NAME, temperature=0, use_responses_api=use_responses_api
    )
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: list = [
        HumanMessage("Sally has green hair, what would her username be?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="foo",
                )
            ],
            tool_choice="required",
        ),
        ToolMessage("sally_green_hair", tool_call_id="foo"),
    ]
    output: AIMessage = cast(AIMessage, llm_with_tool.invoke(msgs))
    assert output.content
    # Should not have called the tool again.
    assert not output.tool_calls and not output.invalid_tool_calls


@pytest.mark.parametrize("use_responses_api", [False])
def test_bind_tools_tool_choice(use_responses_api: bool) -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(
        model=MODEL_NAME, temperature=0, use_responses_api=use_responses_api
    )
    for tool_choice in ("required",):
        llm_with_tools = llm.bind_tools(
            tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice
        )
        msg = cast(AIMessage, llm_with_tools.invoke("you must call a tool"))
        assert len(msg.tool_calls) >= 1

    llm_with_tools = llm.bind_tools(tools=[GenerateUsername, MakeASandwich])
    msg = cast(AIMessage, llm_with_tools.invoke("how are you"))
    assert not msg.tool_calls


@pytest.mark.parametrize("model", [MODEL_NAME])
def test_openai_structured_output(model: str) -> None:
    class MyModel(BaseModel):
        """A Person"""

        name: str
        age: int

    llm = ChatOpenAI(model=model).with_structured_output(MyModel)
    result = llm.invoke("I'm a 27 year old named Erick")
    assert isinstance(result, MyModel)
    assert result.name == "Erick"
    assert result.age == 27


def test_image_token_counting_jpeg() -> None:
    model = ChatOpenAI(model=VISION_MODEL_NAME, temperature=0)
    image_url = "https://sf-maas.s3.us-east-1.amazonaws.com/images/recDq23epr.png"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": "describe this image in detail"},
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    assert expected > 0


@pytest.mark.parametrize("use_responses_api", [False])
def test_tool_calling_strict(use_responses_api: bool) -> None:
    """Test tool calling with strict=True.

    Responses API appears to have fewer constraints on schema when strict=True.
    """

    class magic_function_notrequired_arg(BaseModel):
        """Applies a magic function to an input."""

        input: Optional[int] = Field(default=None)

    model = ChatOpenAI(
        model=MODEL_NAME, temperature=0, use_responses_api=use_responses_api
    )
    # N.B. magic_function adds metadata to schema (min/max for number fields)
    model_with_tools = model.bind_tools([magic_function], strict=True)
    # Having a not-required argument in the schema remains invalid.
    model_with_invalid_tool_schema = model.bind_tools(
        [magic_function_notrequired_arg], strict=True
    )

    # Test invoke
    query = "What is the value of magic_function(3)? Use the tool."
    response = model_with_tools.invoke(query)
    _validate_tool_call_message(response)

    # Test invalid tool schema
    # with pytest.raises(openai.BadRequestError):
    model_with_invalid_tool_schema.invoke(query)

    # Test stream
    full: Optional[BaseMessageChunk] = None
    for chunk in model_with_tools.stream(query):
        full = chunk if full is None else full + chunk  # type: ignore
    assert isinstance(full, AIMessage)
    _validate_tool_call_message(full)

    # Test invalid tool schema
    # with pytest.raises(openai.BadRequestError):
    next(model_with_invalid_tool_schema.stream(query))


@pytest.mark.parametrize("use_responses_api", [False])
@pytest.mark.parametrize(
    ("model", "method"),
    [(MODEL_NAME, "json_schema")],
)
def test_structured_output_strict(
    model: str,
    method: Literal["json_schema"],
    use_responses_api: bool,
) -> None:
    """Test to verify structured output with strict=True."""

    from pydantic import BaseModel as BaseModelProper
    from pydantic import Field as FieldProper

    llm = ChatOpenAI(model=model, use_responses_api=use_responses_api)

    class Joke(BaseModelProper):
        """Joke to tell user."""

        setup: str = FieldProper(description="question to set up a joke")
        punchline: str = FieldProper(description="answer to resolve the joke")

    # Pydantic class
    chat = llm.with_structured_output(Joke, method=method, strict=True)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, Joke)

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, Joke)

    # Schema
    chat = llm.with_structured_output(
        Joke.model_json_schema(), method=method, strict=True
    )
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline"}


@pytest.mark.parametrize("use_responses_api", [False])
@pytest.mark.parametrize(("model", "method"), [(MODEL_NAME, "json_schema")])
def test_nested_structured_output_strict(
    model: str, method: Literal["json_schema"], use_responses_api: bool
) -> None:
    """Test to verify structured output with strict=True for nested object."""

    from typing import TypedDict

    llm = ChatOpenAI(model=model, temperature=0, use_responses_api=use_responses_api)

    class SelfEvaluation(TypedDict):
        score: int
        text: str

    class JokeWithEvaluation(TypedDict):
        """Joke to tell user."""

        setup: str
        punchline: str
        self_evaluation: SelfEvaluation

    # Schema
    chat = llm.with_structured_output(JokeWithEvaluation, method=method, strict=True)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(result["self_evaluation"].keys()) == {"score", "text"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(chunk["self_evaluation"].keys()) == {"score", "text"}


@pytest.mark.parametrize(
    ("strict", "method"),
    [
        (True, "json_schema"),
        (False, "json_schema"),
    ],
)
def test_json_schema_openai_format(
    strict: bool, method: Literal["json_schema"]
) -> None:
    """Test we can pass in OpenAI schema format specifying strict."""
    llm = ChatOpenAI(model=MODEL_NAME)
    schema = {
        "name": "get_weather",
        "description": "Fetches the weather in the given location",
        "strict": strict,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to return the temperature in",
                    "enum": ["F", "C"],
                },
            },
            "additionalProperties": False,
            "required": ["location", "unit"],
        },
    }
    chat = llm.with_structured_output(schema, method=method)
    result = chat.invoke("What is the weather in New York?")
    assert isinstance(result, dict)


@pytest.mark.parametrize("use_responses_api", [False])
def test_stream_o_series(use_responses_api: bool) -> None:
    list(
        ChatOpenAI(model=MODEL_NAME, use_responses_api=use_responses_api).stream(
            "how are you"
        )
    )


@pytest.mark.parametrize("use_responses_api", [False])
async def test_astream_o_series(use_responses_api: bool) -> None:
    async for _ in ChatOpenAI(
        model=MODEL_NAME, use_responses_api=use_responses_api
    ).astream("how are you"):
        pass


class Foo(BaseModel):
    response: str


def test_stream_response_format() -> None:
    full: Optional[BaseMessageChunk] = None
    chunks = []
    for chunk in ChatOpenAI(model=MODEL_NAME).stream("how are ya", response_format=Foo):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]


async def test_astream_response_format() -> None:
    full: Optional[BaseMessageChunk] = None
    chunks = []
    async for chunk in ChatOpenAI(model=MODEL_NAME).astream(
        "how are ya", response_format=Foo
    ):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]


@pytest.mark.skip(reason="This test is temporarily disabled.")
def test_structured_output_and_tools() -> None:
    class ResponseFormat(BaseModel):
        response: str
        explanation: str

    llm = ChatOpenAI(model=MODEL_NAME).bind_tools(
        [GenerateUsername], strict=True, response_format=ResponseFormat
    )

    response = llm.invoke("What weighs more, a pound of feathers or a pound of gold?")
    assert isinstance(response.additional_kwargs["parsed"], ResponseFormat)

    # Test streaming tool calls
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(
        "Generate a user name for Alice, black hair. Use the tool."
    ):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1
    tool_call = full.tool_calls[0]
    assert tool_call["name"] == "GenerateUsername"


def test_tools_and_structured_output() -> None:
    class ResponseFormat(BaseModel):
        response: str
        explanation: str

    llm = ChatOpenAI(model=MODEL_NAME).with_structured_output(
        ResponseFormat,
        strict=True,
        include_raw=True,
        tools=[GenerateUsername],
        tool_choice="required",
    )

    expected_keys = {"raw", "parsing_error", "parsed"}
    query = "Hello"
    tool_query = "Generate a user name for Alice, black hair. Use the tool."
    # Test invoke
    ## Engage structured output
    response = llm.invoke(query)
    assert isinstance(response["parsed"], ResponseFormat)
    ## Engage tool calling
    response_tools = llm.invoke(tool_query)
    ai_msg = response_tools["raw"]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls
    assert isinstance(response["parsed"], ResponseFormat)

    # Test stream
    aggregated: dict = {}
    for chunk in llm.stream(tool_query):
        assert isinstance(chunk, dict)
        assert all(key in expected_keys for key in chunk)
        aggregated = {**aggregated, **chunk}
    assert all(key in aggregated for key in expected_keys)
    assert isinstance(aggregated["raw"], AIMessage)
    assert aggregated["raw"].tool_calls
    assert isinstance(aggregated["parsed"], ResponseFormat)
