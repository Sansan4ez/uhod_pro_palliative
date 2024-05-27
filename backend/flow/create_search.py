"""
Initializes an Qdrant Search with our custom data, using vector search
and semantic ranking.
"""
# from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
# from dotenv import load_dotenv
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.docstore.document import Document

import math
import os

from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.http import models

import re
import tiktoken
from tqdm.auto import tqdm

from gradio_client import Client

embedding_model = os.environ.get("EMBED_MODEL")

DATA_DIR = os.environ.get("DATA_DIR")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME")

def load_and_split_documents() -> list[dict]:
    """
    Loads our documents from disc and split them into chunks.
    Returns a list of dictionaries.
    """
    # loader = DirectoryLoader(
    #     DATA_DIR, loader_cls=UnstructuredMarkdownLoader, show_progress=True
    # )
    # docs = loader.load()
    # print(f"loaded {len(docs)} documents")

    with open(f'{DATA_DIR}/bz.md', 'r', encoding='utf-8') as file:
            text = file.read()

    # Удалим ключевые слова, с учетом многострочного режима
    pattern = r"Ключевые слова.*(\n|$)"

    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)

    headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    split_docs = markdown_splitter.split_text(cleaned_text)

    # Добавление номера основного чанка в метаданные
    for i, doc in enumerate(split_docs):
        doc.metadata["parent_id"] = f'{i}'

    # Выделение вопросов, саммари, контента и 'Header3' из чанка в отдельные чанки
    cleaned_docs = []
    question_docs = []
    summary_docs = []
    header_docs = []
    docs = split_docs
    for i, doc in enumerate(docs):
        # выделяем заголовок 3-го уровня для добавления в чанк
        header_3 = doc.metadata.get('Header3')
        # выделяем только чанки
        tolko_chunk = re.sub(r'~~~.*?~~~', '', doc.page_content, flags=re.DOTALL)
        cleaned_chunk = tolko_chunk.lstrip()        # удаляем лишние пробелы в начале строки
        if header_3:
            doc.metadata['content'] = f'{header_3}/n{cleaned_chunk}'        # добавляем сам чанк в метаданные
        else:
            doc.metadata['content'] = f'{cleaned_chunk}'
        cleaned_doc = Document(page_content=cleaned_chunk, metadata=doc.metadata.copy())
        cleaned_docs.append(cleaned_doc)


        # Регулярное выражение для поиска раздела "Вопросы:" и последующих вопросов
        # Используем негативный просмотр вперед, чтобы остановиться на следующем разделе или конце текста
        pattern = r"Вопросы:\n(.+?)(?:\n\n|\Z)"

        # Ищем весь блок вопросов, используя флаги DOTALL и MULTILINE для многострочного поиска
        match = re.search(pattern, doc.page_content, flags=re.DOTALL)

        if match:
            # Извлекаем весь блок вопросов
            questions_block = match.group(1)

            # Ищем все вопросы в блоке. Вопросы могут начинаться с "-", цифры, или без специального символа,
            # но должны заканчиваться знаком вопроса.
            questions = re.findall(r"(?m)^\s*([-\d]?.*?\?)", questions_block)

            # Выводим список вопросов
            for question in questions:
                cleaned_question = re.sub(r"^\s*[\d\-\.\(\)]*\s*", "", question)
                # Создание нового объекта Document для каждого вопроса
                question_doc = Document(page_content=cleaned_question, metadata=doc.metadata.copy())
                question_docs.append(question_doc)

        # Измененное регулярное выражение для поиска текста "Саммари" до символов "~~~"
        summary_match = re.search(r"Саммари:\s*(.*?)(?=~~~)", doc.page_content, flags=re.DOTALL)

        if summary_match:
            summary = summary_match.group(1).strip()
            # Создание нового объекта Document для каждого саммари
            summary_doc = Document(page_content=summary, metadata=doc.metadata.copy())
            summary_docs.append(summary_doc)

        # Выделяем чанки с заголовками 'Header3', используем .get() для безопасного доступа
        # Перебираем заголовки в порядке приоритета, если 'Header3' отсутствует
        for header_key in ['Header3', 'Header2', 'Header1']:
            header_chunk = doc.metadata.get(header_key)
            if header_chunk:  # Если найден непустой заголовок
                # Создаем и добавляем новый Document
                header_doc = Document(page_content=header_chunk, metadata=doc.metadata.copy())
                header_docs.append(header_doc)
                break  # Прерываем цикл после добавления первого найденного заголовка

    # Объединяем все документы в один список
    combined_docs = cleaned_docs + question_docs + summary_docs + header_docs


    # Если чанки слишком большие, их можно поделить еще на подчанки
    # Инициализация сплиттера
    MaxChankSize = 250
    RCTsplitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=MaxChankSize,
        chunk_overlap=int(MaxChankSize * 0.2)
    )
    # Итоговый список документов
    documents = []

    for document in combined_docs:
        if len(document.page_content) > MaxChankSize:
            # Разделение контекста на подчанки
            subchunks = RCTsplitter.split_text(document.page_content)
            Nsub=len(subchunks)
            for i, subchunk in enumerate(subchunks):
                # Создание нового объекта Document для каждого подчанка
                subchunk_doc = Document(page_content=subchunk, metadata=document.metadata.copy())
                # Добавление номера подчанка в метаданные
                subchunk_doc.metadata["subchunk"] = f'{i+1}/{Nsub}'
                documents.append(subchunk_doc)
        else:
            # Добавление неизмененного чанка в итоговый список
            documents.append(document)

    print(f"split into {len(documents)} documents")

    final_docs = []
    for i, doc in enumerate(documents):
        doc.metadata["page_content"] = f"{doc.page_content}"

        doc_dict = {
            "id": str(i),       #??? почему "id" - строка, а не целое число?
            "page_content": f"{doc.page_content}",
            "payload": doc.metadata,
        }
        final_docs.append(doc_dict)

    return final_docs

def initialize(search_client):
    """
    Initializes an Qdrant Search with our custom data, using vector
    search.
    """
    docs = load_and_split_documents()

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    token_sizes = [len(encoding.encode(doc["page_content"])) for doc in docs]
    batch_size = 50
    num_batches = math.ceil(len(docs) / batch_size)

    print(f"embedding {len(docs)} documents in {num_batches} batches of {batch_size}, using OpenAI embeddings")
    print(f"Total tokens: {sum(token_sizes)}, average token: {int(sum(token_sizes) / len(token_sizes))}")

    client = OpenAI()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(docs))
        batch_docs = docs[start_idx:end_idx]
        response = client.embeddings.create(
            input=[doc["page_content"] for doc in batch_docs],
            model=embedding_model,
        )

        my_embeddings = response.data

        for j, doc in enumerate(batch_docs):
            doc["embedding"] = my_embeddings[j].embedding

    print(f"batch {i+1} embedding success")

    def compute_vector(text):
        query_json = {
        "inputs": text
        }
        client3 = Client("http://sparse_vector:7860")
        indices, values = client3.predict(
            question=query_json,
            api_name="/predict"
        )

        return indices, values

    # Use recreate_collection if you are experimenting and running the script several times.
    # This function will first try to remove an existing collection with the same name.
    # !!! Instead, to create a brand new collection that cannot be recreated, use the client.create_collection() method.

    search_client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            "text_dense": models.VectorParams(
                size=len(docs[0]["embedding"]),
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "text_sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            ),
        },
    )

    # Upload data to collection

    ids = [int(doc["id"]) for doc in docs]
    dense_vectors = [doc["embedding"] for doc in docs]
    payloads = [doc["payload"] for doc in docs]

    content=[doc["page_content"] for doc in docs]
    doc_indices = []
    doc_values = []
    for i in range(len(content)):
        doc_indice, doc_value = compute_vector(content[i])
        doc_indices.append(doc_indice)
        doc_values.append(doc_value)

    sparse_vectors = []
    for i in range(len(doc_indices)):
        sparse_vector = models.SparseVector(
            indices=doc_indices[i],
            values=doc_values[i]
            )
        sparse_vectors.append(sparse_vector)

    def split_batches(ids, dense_vectors, sparse_vectors, payloads, max_size=2):
        # Split the data into batches
        # This is a simplified logic. You will need to calculate the batch size based on actual payload size
        # `max_size` - количество записей (points) в одном batch
        batches = []
        current_batch = {"ids": [], "dense_vectors": [], "sparse_vectors": [], "payloads": []}
        for id, dense_vector, sparse_vector, payload in zip(ids, dense_vectors, sparse_vectors, payloads):
            # Logic to add data to current_batch or start a new batch based on max_size
            # This is pseudo-code; actual implementation will depend on how you calculate size
            if len(current_batch["ids"]) < max_size:
                current_batch["ids"].append(id)
                current_batch["dense_vectors"].append(dense_vector)
                current_batch["sparse_vectors"].append(sparse_vector)
                current_batch["payloads"].append(payload)
            else:
                batches.append(current_batch)
                current_batch = {"ids": [id], "dense_vectors": [dense_vector], "sparse_vectors": [sparse_vector], "payloads": [payload]}
        # Don't forget to add the last batch if it's not empty
        if current_batch["ids"]:
            batches.append(current_batch)
        return batches

    # Example usage (you need to define `max_size` based on your needs)
    batches = split_batches(ids, dense_vectors, sparse_vectors, payloads, max_size=40)  # Adjust `max_size` appropriately

    for batch in tqdm(batches):
        search_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=models.Batch(
                ids=batch["ids"],
                vectors={
                        "text_dense": batch["dense_vectors"],
                        "text_sparse": batch["sparse_vectors"],
                },
                payloads=batch["payloads"],
            ),
        )

def delete(search_client):
    """
    Deletes the Qdrant collection.
    """
    print(f"deleting collection {QDRANT_COLLECTION_NAME}")
    search_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

def main():
    search_client = QdrantClient("qdrant_vb", port=6333)
    delete(search_client)
    initialize(search_client)

if __name__ == "__main__":
    main()