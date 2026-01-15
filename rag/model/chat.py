import os
from openai import OpenAI
from tiktoken import get_encoding

from rag.model.embedding import VectorStore
from rag.preprocess.read import ReadFiles

DEFAULT_CHAT_TEMPLATE = \
"""
下面会给出{cnt}条参考信息，可能与用户的提问相关，也可能无关。
如果相关，则严格按照参考信息回答用户提问；
如果不相关，则按照自身的知识进行回答。
用户提问
{question}
参考信息
'''
{reference}
'''
"""



class OpenAIChatModel:
    def __init__(self, model: str, embedding: str=None, base_url=None, api_key=None):
        self.model = model
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.vector_store = None
        if embedding is not None:
            self.vector_store = VectorStore(embedding, self.client)

    def set_knowledge_source(self, src):
        read_files = ReadFiles(path=src)
        chunks = read_files.get_content()
        self.vector_store.init_store(chunks)

    def set_vector_store(self, vector_store: VectorStore):
        self.vector_store = vector_store


    def chat(self, query, top_k=1):
        _, top_k_text = self.vector_store.query(query, top_k=top_k)
        context = [f"{i+1}.\n{text}\n" for i, text in enumerate(top_k_text)]
        context = "".join(context)
        messages = [{
            "role": "user",
            "content": DEFAULT_CHAT_TEMPLATE.format(cnt=top_k, question=query, reference=context)
        }]
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content



