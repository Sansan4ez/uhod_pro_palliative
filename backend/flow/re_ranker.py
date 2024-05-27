from promptflow import tool
from gradio_client import Client

# from dotenv import load_dotenv
# load_dotenv()

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def re_rank(query: str, docs: list, limit_chunks=3) -> list:

    # Ранжирование всех результатов с помощью кросс-энкодера
    pairs = []
    for doc in docs:
        pairs.append([query, doc['payload']['content']])

    client = Client("http://reranker:7860")
    data = { "inputs": pairs }
    cross_scores = client.predict(
            data=data,
            api_name="/predict"
    )

    # добавим в словарь с результатами поиска cross-score
    for idx in range(len(cross_scores)):
        docs[idx]["cross_score"] = cross_scores[idx]
    # отсортируем результаты по полю cross_score
    results_sorted = sorted(docs, key=lambda x: x["cross_score"], reverse=True)

    # возьмем три - пять лучших результатов
    re_docs = [
        {
            "id": doc["id"],
            "score": f'{doc["score"]:.2f}',
            "cross_score" : f'{float(doc["cross_score"]):.2f}',
            "content": doc['payload']['content']
        } for doc in results_sorted[0:limit_chunks]]

    return re_docs
