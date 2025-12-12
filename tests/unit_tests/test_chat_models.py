"""Test chat model integration."""


import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_siliconflow.chat_models import ChatSiliconFlow


def test_chat_siliconflow() -> None:
    """Test ChatSiliconFlow wrapper."""
    chat = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3.1", temperature=0)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_siliconflow_system_message() -> None:
    """Test ChatSiliconFlow wrapper with system message."""
    chat = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3.1", temperature=0)
    system_message = SystemMessage(content="You are a helpful assistant.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.asyncio
async def test_async_chat_siliconflow() -> None:
    """Test async ChatSiliconFlow wrapper."""
    chat = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3.1", temperature=0)
    message = HumanMessage(content="Hello")
    response = await chat.ainvoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

def test_async_chat_siliconflow_stream_usage() -> None:
    """
    Integration test checking that ChatSiliconFlow correctly handles
    cumulative usage stats from the real SiliconFlow API.
    """
    # 1. Setup
    # Ensure SILICONFLOW_API_KEY is in your environment
    llm = ChatSiliconFlow(
        model="deepseek-ai/DeepSeek-V3",  # Use a cheap/fast model
        temperature=0,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt = "Say 'hello from SiliconFlow!'"
    chunks = []

    # 2. Stream and collect
    for chunk in llm.stream(prompt):
        chunks.append(chunk)

    # 3. Verify Basic Response
    assert len(chunks) > 0
    full_content = "".join(c.content for c in chunks)
    assert "hello " in full_content.lower()

    # 4. Verify Usage Logic (The Core Test)

    # Filter only chunks that actually came with metadata
    usage_chunks = [c for c in chunks if c.usage_metadata]
    print(f"\n--- Usage Chunks Debug Info ---")
    for c in usage_chunks:
        print(f"Chunk usage metadata: {c.usage_metadata}")
    assert len(usage_chunks) > 0, "No usage metadata received from API"

    # Calculate Aggregates (simulate LangChain's final callback aggregation)
    total_input_calculated = sum(c.usage_metadata["input_tokens"] for c in usage_chunks)
    total_output_calculated = sum(c.usage_metadata["output_tokens"] for c in usage_chunks)

    print(f"\n--- Debug Info ---")
    print(f"Chunks received: {len(chunks)}")
    print(f"Calculated Input: {total_input_calculated}")
    print(f"Calculated Output: {total_output_calculated}")

    # Assertion A: Input Tokens (Prevent Overcounting)
    assert 0 < total_input_calculated < 20, (
        f"Input tokens suspiciously high ({total_input_calculated}). "
        "Delta logic might be failing to zero out subsequent chunks."
    )

    # Assertion B: Input Tokens Distribution
    # If the fix works, only the first usage-containing chunk should have input tokens.
    # Subsequent chunks should have 0 input tokens (since cumulative total didn't change).
    if len(usage_chunks) > 1:
        first_input = usage_chunks[0].usage_metadata["input_tokens"]
        last_input = usage_chunks[-1].usage_metadata["input_tokens"]
        assert first_input > 0
        assert last_input == 0, "Subsequent chunks should have 0 input tokens (delta)"

    # Assertion C: Output Tokens (Prevent Overcounting)
    # Each chunk should contribute some output tokens, but not cumulatively.
    assert total_output_calculated > 0
    assert total_output_calculated < len(usage_chunks) * 2

@pytest.mark.asyncio
async def test_async_chat_siliconflow_astream_usage() -> None:
    """
    Test checking that ChatSiliconFlow correctly handles cumulative usage stats
    during ASYNC streaming (astream).
    """
    llm = ChatSiliconFlow(
        model="deepseek-ai/DeepSeek-V3",
        temperature=0,
        model_kwargs={"stream_options": {"include_usage": True}}
    )

    prompt = "Say 'test'"
    chunks = []

    # Using astream
    async for chunk in llm.astream(prompt):
        chunks.append(chunk)

    # Filter chunks with usage
    usage_chunks = [c for c in chunks if c.usage_metadata]
    assert len(usage_chunks) > 0, "No usage metadata received from API"

    # Calculate Aggregates
    total_input = sum(c.usage_metadata["input_tokens"] for c in usage_chunks)
    total_output = sum(c.usage_metadata["output_tokens"] for c in usage_chunks)

    print(f"\n--- Async Debug Info ---")
    print(f"Chunks: {len(chunks)}, Usage Chunks: {len(usage_chunks)}")
    print(f"Calculated Input: {total_input}")
    print(f"Calculated Output: {total_output}")

    # Assertion A: Input Tokens
    assert 0 < total_input < 10, f"Async Input tokens suspiciously high ({total_input})"

    # Assertion B: Input Delta Logic
    if len(usage_chunks) > 1:
        assert usage_chunks[0].usage_metadata["input_tokens"] > 0
        assert usage_chunks[-1].usage_metadata["input_tokens"] == 0

    # Assertion C: Output Tokens
    assert 0 < total_output < len(usage_chunks) * 2
