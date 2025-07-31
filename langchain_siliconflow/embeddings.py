from typing import Any, Dict, List, Optional

import openai
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, model_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class SiliconFlowEmbeddings(BaseModel, Embeddings):
    """SiliconFlow embedding model integration.

    Setup:
        Install ``langchain-siliconflow`` and set environment variable
        ``SILICONFLOW_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-siliconflow
            export SILICONFLOW_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of SiliconFlow model to use. For example, `Qwen/Qwen3-Embedding-8B`.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_siliconflow import SiliconFlowEmbeddings

            embed = SiliconFlowEmbeddings(
                model="Qwen/Qwen3-Embedding-8B",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)

            # multiple:
            await embed.aembed_documents(input_texts)

    """

    model: str
    """The name of the model"""
    timeout: Optional[int] = None
    max_retries: int = 2

    siliconflow_api_key: Optional[SecretStr] = None

    client: Any
    async_client: Any

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        values["siliconflow_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "siliconflow_api_key",
                "SILICONFLOW_API_KEY",
            )
        )
        api_key = values["siliconflow_api_key"].get_secret_value()
        values["client"] = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.com/v1",
        )
        values["async_client"] = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.com/v1",
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [r.embedding for r in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        response = await self.async_client.embeddings.create(
            model=self.model, input=texts
        )
        return [r.embedding for r in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        response = await self.async_client.embeddings.create(
            model=self.model, input=[text]
        )
        return response.data[0].embedding
