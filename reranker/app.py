import gradio as gr
from sentence_transformers import CrossEncoder

model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder = CrossEncoder(model_id, max_length=512)

def ranker(data):
    # Извлекаем 'inputs' из полученного JSON
    inputs = data.get('inputs', [])
    # Проверяем, что inputs это список
    if not isinstance(inputs, list):
        return {"error": "Неправильный формат входных данных. Ожидается список пар предложений."}

    # Предполагаем, что inputs - это список пар предложений
    scores = cross_encoder.predict(inputs)
    return scores

iface = gr.Interface(
    fn=ranker,
    inputs=gr.JSON(label="Введите JSON в формате {'inputs': [['sentence1', 'sentence2'], ...]}"),
    outputs="json",
    title="Rerank sequence",
    description="Rerank any sequence using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model under the hood!"
)

iface.launch()