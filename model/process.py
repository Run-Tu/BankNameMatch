import json
import pandas as pd

def process_query_file(input_file, output_file="./data/query.jsonl"):
    """
    处理Excel文件并将指定列转换为JSON Lines格式。

    Args:
        input_file (str): 输入的Excel文件路径。
        output_file (str): 输出的JSON Lines文件路径。
    """
    # 读取Excel文件
    schema_df = pd.read_excel(input_file)

    schema_jsonl = []
    for _,row in schema_df.iterrows():
        schema_jsonl.append({"query":row["对方行名"]})

    with open("./data/query.jsonl", 'w', encoding="UTF-8") as file:
        for item in schema_jsonl:
            file.write(json.dumps(item,ensure_ascii=False) + '\n')

def load_jsonl(file_path):
    """
    加载JSON Lines格式的文件。

    Args:
        file_path (str): JSON Lines文件路径。

    Returns:
        list: 包含文件内容的列表。
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data