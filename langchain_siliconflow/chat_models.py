"""SiliconFlow chat models."""

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from typing import Union


def _create_chat_result(
    message: BaseMessage,
    usage: Union[dict, None],
    model_name: str
) -> ChatResult:
    """Create a ChatResult with standardized metadata."""
    if usage:
        message.usage_metadata = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_token_details": {},
            "output_token_details": {},
            "model_name": model_name
        }
    message.response_metadata = {"model_name": model_name}
    return ChatResult(
        generations=[ChatGeneration(message=message)],
        response_metadata={"model_name": model_name}
    )

from pydantic import Field, SecretStr, model_validator

from langchain_siliconflow.utils import validate_environment


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown message type: {message}")

    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: dict) -> AIMessage:
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))
    elif role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""), tool_call_id=_dict.get("tool_call_id")
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)


class ChatSiliconFlow(BaseChatModel):
    """SiliconFlow chat model integration.

    Setup:
        Install ``langchain-siliconflow`` and set environment variable ``SILICONFLOW_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-siliconflow
            export SILICONFLOW_API_KEY="your-api-key"
    """

    model_name: str = Field(alias="model")
    """The name of the model"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    siliconflow_api_key: Optional[SecretStr] = None

    client: openai.OpenAI
    async_client: openai.AsyncOpenAI

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        return validate_environment(values)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-siliconflow"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            **kwargs,
        )
        message = _convert_dict_to_message(response.choices[0].message.model_dump())
        return _create_chat_result(message, response.usage, self.model_name)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        first_chunk = True
        for chunk in self.client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            stream=True,
            **kwargs,
        ):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            msg = AIMessageChunk(content=delta.content)
            if first_chunk and chunk.usage:
                msg.usage_metadata = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                    "model_name": self.model_name
                }
            msg.response_metadata = {"model_name": self.model_name}
            yield ChatGenerationChunk(message=msg)
            first_chunk = False

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            **kwargs,
        )
        message = _convert_dict_to_message(response.choices[0].message.model_dump())
        return _create_chat_result(message, response.usage, self.model_name)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        first_chunk = True
        async for chunk in await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=message_dicts,
            stream=True,
            **kwargs,
        ):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            msg = AIMessageChunk(content=delta.content)
            if first_chunk and chunk.usage:
                msg.usage_metadata = {
                    "input_tokens": chunk.usage.prompt_tokens,
                    "output_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                    "model_name": self.model_name
                }
            msg.response_metadata = {"model_name": self.model_name}
            yield ChatGenerationChunk(message=msg)
            first_chunk = False
