import os
import uuid
import logging
import pandas as pd
from fastapi import FastAPI, Form
from concurrent.futures import ThreadPoolExecutor
from model.process import process_query_file, load_jsonl
from model.search import index, search
from model.models import FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=10)
# 全局变量
model = None
faiss_index = None
# 用于存储文件状态的字典
file_status = {}

def generate_unique_id():
    return uuid.uuid4().hex[:8]


def initialize_model():
    """初始化模型"""
    logger.info("Loading model...")
    return FlagModel(
        model_name_or_path="/root/autodl-tmp/data_match/model/models--TidalTail--FinQA-FlagEmbedding/snapshots/272edad9ab6cd0160d2baf3597f881c8d49f1de1",
        query_instruction_for_retrieval=None,
        use_fp16=False
    )


def build_faiss_index(model):
    """构建 Faiss 索引"""
    logger.info("Building Faiss index...")
    return index(
        model=model,
        index_factory="Flat",
        save_path="/root/autodl-tmp/data_match/20240509_embeddings.memmap"
    )


def update_file_status(unique_id, status, result_file=None):
    """更新文件状态"""
    file_status[unique_id] = {
        "status":status,
        "result_file":result_file
    }


def save_results_to_excel(query_df, query_schema, output_file_result):
    """保存结果到 Excel 文件"""
    query_df["TT_match_result"] = query_df["对方行名"].map(query_schema)
    query_df.to_excel(output_file_result, index=False)


def process_and_search_task(unique_id, query_file_path, output_file_result):
    """后台处理文件任务"""
    global model, faiss_index
    update_file_status(unique_id, "processing")
    if not os.path.exists(query_file_path):
        update_file_status(unique_id, "error")
        return {"error": "Query file not found."}
    try:
        process_query_file(query_file_path)
        query = load_jsonl("./data/query.jsonl")
        schema = load_jsonl("./data/schema.jsonl")
        scores, indices = search(
            model=model,
            queries=query,
            faiss_index=faiss_index,
            k=1,
            batch_size=1,
            max_length=32
        )
        retrieval_results = [
            [schema[idx]["schema"] for idx in index_array if idx != -1]
            for index_array in indices
        ]

        query_schema = {
            query_item['query']: match_schema[0] if match_schema else None
            for query_item, match_schema in zip(query, retrieval_results)
        }

        query_df = pd.read_excel(query_file_path)
        save_results_to_excel(query_df, query_schema, output_file_result)
        update_file_status(unique_id, "completed", output_file_result)
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        update_file_status(unique_id, "error")


@app.on_event("startup")
def load_model_and_index():
    global model, faiss_index
    model = initialize_model()
    faiss_index = build_faiss_index(model)


@app.post("/v1/process_and_search/")
async def process_and_search(query_file_path: str = Form(...)):
    unique_id = generate_unique_id()
    output_file_result = f"./data/result_{unique_id}.xlsx"
    executor.submit(process_and_search_task, unique_id, query_file_path, output_file_result)
    return {"unique_id": unique_id}


@app.get("/check_status/")
async def check_status(unique_id: str):
    status_info = file_status.get(unique_id, {"status": "not_found"})
    if status_info["status"] == "completed":
        return {
            "unique_id": unique_id,
            "status": status_info["status"],
            "result_file": status_info["result_file"]
        }
    
    return {
        "unique_id": unique_id,
        "status": status_info["status"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)