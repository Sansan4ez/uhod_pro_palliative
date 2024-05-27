from promptflow import tool
from gradio_client import Client

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def compute_vector(question: str):
    query_json = {
    "inputs": question
    }
    client2 = Client("http://sparse_vector:7860")
    indices, values = client2.predict(
        question=query_json,
        api_name="/predict"
    )

    return indices, values
