from importlib import metadata

from langchain_siliconflow.chat_models import ChatSiliconFlow
from langchain_siliconflow.document_loaders import SiliconFlowLoader
from langchain_siliconflow.embeddings import SiliconFlowEmbeddings
from langchain_siliconflow.retrievers import SiliconFlowRetriever
from langchain_siliconflow.toolkits import SiliconFlowToolkit
from langchain_siliconflow.tools import SiliconFlowTool
from langchain_siliconflow.vectorstores import SiliconFlowVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatSiliconFlow",
    "SiliconFlowVectorStore",
    "SiliconFlowEmbeddings",
    "SiliconFlowLoader",
    "SiliconFlowRetriever",
    "SiliconFlowToolkit",
    "SiliconFlowTool",
    "__version__",
]
