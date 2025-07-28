# langchain-siliconflow

This package contains the LangChain integration with SiliconFlow

## Installation

```bash
pip install -U langchain-siliconflow
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSiliconFlow` class exposes chat models from SiliconFlow.

```python
from langchain_siliconflow import ChatSiliconFlow

llm = ChatSiliconFlow()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SiliconFlowEmbeddings` class exposes embeddings from SiliconFlow.

```python
from langchain_siliconflow import SiliconFlowEmbeddings

embeddings = SiliconFlowEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SiliconFlowLLM` class exposes LLMs from SiliconFlow.

```python
from langchain_siliconflow import SiliconFlowLLM

llm = SiliconFlowLLM()
llm.invoke("The meaning of life is")
```
