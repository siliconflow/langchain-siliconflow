from typing import Type

from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

from langchain_siliconflow.retrievers import SiliconFlowRetriever


class TestSiliconFlowRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[SiliconFlowRetriever]:
        """Get an empty vectorstore for unit tests."""
        return SiliconFlowRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
