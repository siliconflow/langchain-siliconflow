from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, SecretStr, model_validator

from langchain_siliconflow.utils import validate_environment


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
        return validate_environment(values)

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
