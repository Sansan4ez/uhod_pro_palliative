import datetime
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str) -> str:
    # Generate UUID
    current_datetime = datetime.datetime.now()
    id = int(f"{current_datetime.strftime('%Y%m%d%H%M%S')}{current_datetime.microsecond}")
    return id
