# import faiss
# import numpy as np
# from backend.embedding import get_embeddings

# class VectorStore:
#     def __init__(self):
#         self.index = None
#         self.texts = []


#     def add_documents(self, docs):
#         embeddings = get_embeddings(docs)
#         self.texts.extend(docs)


#         if self.index is None:
#             self.index = faiss.IndexFlatL2(len(embeddings[0]))

#         self.index.add(np.array(embeddings))

#     def search(self, query, k=3):
#         if self.index is None:
#             return []


#         query_embedding = get_embeddings([query])
#         distances, indices = self.index.search(np.array(query_embedding), k)

#         results = []
#         for i, dist in zip(indices[0], distances[0]):
#             if dist < 1.5:  # threshold (can tune later)
#                 results.append(self.texts[i])

#         return results 

import faiss
import numpy as np
from backend.embedding import get_embeddings

# 🔹 Re-ranking function
def rerank(query, results):
    query_words = set(query.lower().split())
    
    scored = []
    for text in results:
        score = sum(1 for word in query_words if word in text.lower())
        scored.append((score, text))
    
    scored.sort(reverse=True)
    return [text for _, text in scored]


# 🔹 Entity extraction (MOVED OUTSIDE CLASS)
def extract_entities(text):
    return list(set(text.lower().split()))


class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadata = [] # new for metadata
        self.knowledge_graph = {}

    def add_documents(self, docs):
        embeddings = get_embeddings(docs)
        self.texts.extend(docs)

        # new metadata handling
        for i, doc in enumerate(docs):
          self.metadata.append({"id": i, "length": len(doc)})

        #   old code


        # ✅ Knowledge graph building (FIXED)
        for doc in docs:
            entities = extract_entities(doc)
            for e in entities:
                if e not in self.knowledge_graph:
                    self.knowledge_graph[e] = []
                self.knowledge_graph[e].append(doc)

        # FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))

        self.index.add(np.array(embeddings))

    def search(self, query, k=3):
        if self.index is None:
            return []

        # FAISS search
        query_embedding = get_embeddings([query])
        distances, indices = self.index.search(np.array(query_embedding), k)

# old code

        # results = []
        # for i, dist in zip(indices[0], distances[0]):
        #     if dist < 1.5:
        #         results.append(self.texts[i])

        # new code

        filtered_results = []
        for i, dist in zip(indices[0], distances[0]):
           if dist < 1.5 and len(self.texts[i]) > 20:
             filtered_results.append(self.texts[i])

        results = filtered_results

# old code

        # ✅ Knowledge graph boost (AFTER results created)
        for word in query.lower().split():
            if word in self.knowledge_graph:
                results.extend(self.knowledge_graph[word])

        # ✅ Re-ranking
        results = rerank(query, results)

        return results