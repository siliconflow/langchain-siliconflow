"""Test SiliconFlow embeddings."""

from typing import Type

from langchain_siliconflow.embeddings import SiliconFlowEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SiliconFlowEmbeddings]:
        return SiliconFlowEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
