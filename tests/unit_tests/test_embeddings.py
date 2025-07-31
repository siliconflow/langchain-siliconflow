"""Test embedding model integration."""

from typing import Type

from langchain_siliconflow.embeddings import SiliconFlowEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestSiliconFlowEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[SiliconFlowEmbeddings]:
        return SiliconFlowEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
