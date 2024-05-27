import os

from typing import List
from promptflow import tool

# import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# import json
from dotenv import load_dotenv
load_dotenv()

QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

@tool
def retrieve_knowledge(sparse_vectors: List, embedding: List[float], limit_dense=3, limit_sparse=3) -> str:

  search_client = QdrantClient("qdrant_vb", port=6333)

  query_dense_vector = embedding

  indices = sparse_vectors[0]
  values = sparse_vectors[1]
  query_sparse_vector=models.SparseVector(
      indices=indices,
      values=values,
  )

  results = search_client.search_batch(
    collection_name=QDRANT_COLLECTION_NAME,
    requests=[
      models.SearchRequest(
        vector=models.NamedVector(
            name="text_dense",
            vector=query_dense_vector,
        ),
        limit=int(limit_dense),
        with_payload=True,
        # with_vectors=False,   # True/False - отображение векторов - the default is False
      ),
      models.SearchRequest(
        vector=models.NamedSparseVector(
            name="text_sparse",
            vector=query_sparse_vector,
        ),
        limit=int(limit_sparse),
        with_payload=True,
        # with_vectors=False,   # True/False - отображение векторов - the default is False
      ),
    ],
  )

  result_list = []
  for result in results:
    for res in result:
      result_list.append(res)

  # Отбор уникальных чанков
  unique_contents = set()
  unique_docs = []
  for doc in result_list:
    if doc.payload['parent_id'] not in unique_contents:    # сравниваем по номеру 'cleaned_chunk'
      unique_docs.append(doc)
      unique_contents.add(doc.payload['parent_id'])
  unique_contents = list(unique_contents)
  unique_docs = list(unique_docs)

  # сделаем список из словарей с результатами поиска
  docs = [
    {
      "id": str(doc.id),
      "score": float(doc.score),
      "payload": doc.payload
    } for doc in unique_docs]

  return docs
