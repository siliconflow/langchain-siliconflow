"""Test chat model integration."""

from typing import Type

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


@pytest.mark.asyncio
async def test_async_chat_siliconflow_streaming() -> None:
    """Test async streaming ChatSiliconFlow wrapper."""
    chat = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3.1", temperature=0)
    message = HumanMessage(content="Hello")
    response = chat.astream([message])
    async for chunk in response:
        assert isinstance(chunk.content, str)
