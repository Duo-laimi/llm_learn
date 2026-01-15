import os
from typing import Optional, Union, List

from openai import OpenAI
import numpy as np
import pickle as pkl


class VectorStore:
    def __init__(self, model, client=None, base_url=None, api_key=None):
        self.model = model
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.vectors: Optional[np.ndarray] = None
        self.contents: Optional[List[str]] = None

    def init_store(self, text: Union[str, List[str]]):
        self.vectors = self.get_vectors(text)
        self.contents = text

    def get_vectors(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]
        embeds = []
        for content in text:
            response = self.client.embeddings.create(input=content, model=self.model)
            embed = response.data[0].embedding
            embeds.append(embed)
        embeds = np.asarray(embeds, dtype=np.float32)
        return embeds

    def persist(self, name, path=None):
        vector_path = f"{name}.npy"
        text_path = f"{name}.txt"
        if path is not None:
            vector_path = os.path.join(path, vector_path)
            text_path = os.path.join(path, text_path)
        np.save(vector_path, self.vectors, allow_pickle=True)
        with open(text_path, "wb") as file:
            pkl.dump(self.contents, file)

    def load_vectors(self, name, path=None):
        vector_path = f"{name}.npy"
        text_path = f"{name}.txt"
        if path is not None:
            vector_path = os.path.join(path, vector_path)
            text_path = os.path.join(path, text_path)
        self.vectors = np.load(vector_path)
        with open(text_path, "rb") as file:
            self.contents = pkl.load(file)

    @staticmethod
    def get_similarity(vector1: np.ndarray, vector2: np.ndarray):
        if vector1.ndim == 1:
            vector1 = np.expand_dims(vector1, axis=0)
        if vector2.ndim == 1:
            vector2 = np.expand_dims(vector2, axis=0)
        norm1 = np.linalg.norm(vector1, axis=-1, keepdims=True)
        norm2 = np.linalg.norm(vector2, axis=-1, keepdims=True)
        norm = norm1 @ norm2.T
        dot = vector1 @ vector2.T
        return dot / norm

    def query(self, text, top_k=1):
        query_embed = self.get_vectors(text)
        cos = self.get_similarity(query_embed, self.vectors).flatten()
        topk_idx = np.argsort(-cos)[:top_k]
        vectors = self.vectors[topk_idx]
        texts = []
        for idx in topk_idx:
            texts.append(self.contents[idx])
        return vectors, texts




