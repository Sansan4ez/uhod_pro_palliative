import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_id = "naver/ecir23-scratch-tydi-russian-splade"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

def compute_vector(question: str):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    Args:
    logits (torch.Tensor): The logits output from a model.
    attention_mask (torch.Tensor): The attention mask corresponding to the input tokens.
    Returns:
    torch.Tensor: Computed vector.
    """
    # Извлекаем 'inputs' из полученного JSON
    text = question.get('inputs', '')
    # Проверяем, что text это строка
    if not isinstance(text, str):
        return {"error": "Неправильный формат входных данных. Ожидается строка."}

    tokens = tokenizer(text, truncation=True, padding=True, max_length=2048, return_tensors="pt")
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()
    indices = vec.nonzero().numpy().flatten().tolist()
    values = vec.detach().numpy()[indices].tolist()

    return indices, values

sparse_int = gr.Interface(
    fn=compute_vector,
    inputs=gr.JSON(label="Введите JSON в формате {'inputs': 'sentence1'}"),
    outputs="json",
    title="Sparse vector",
    description="Compute sparse vector using the 'naver/ecir23-scratch-tydi-russian-splade' model under the hood!"
)

sparse_int.launch()