"""Test SiliconFlow embeddings."""

from typing import Type

from langchain_siliconflow.embeddings import SiliconFlowEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestSiliconFlowEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SiliconFlowEmbeddings]:
        return SiliconFlowEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "Qwen/Qwen3-Embedding-8B"}
