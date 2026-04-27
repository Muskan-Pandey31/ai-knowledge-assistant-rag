import faiss
import numpy as np
from backend.embedding import get_embeddings

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def add_documents(self, docs):
        embeddings = get_embeddings(docs)
        self.texts.extend(docs)

        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))

        self.index.add(np.array(embeddings))

    def search(self, query, k=3):
        if self.index is None:
            return []

        query_embedding = get_embeddings([query])
        distances, indices = self.index.search(np.array(query_embedding), k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if dist < 1.5:  # threshold (can tune later)
                results.append(self.texts[i])

        return results