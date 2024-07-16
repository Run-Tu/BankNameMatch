import os
import uuid
import logging
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from model.process import process_query_file, load_jsonl
from model.search import index, search
from model.models import FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# 添加CORS中间件解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，如果你有特定的前端网址，可以指定它们
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头信息
)

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
async def upload(query_file: UploadFile = File(...)):
    unique_id = generate_unique_id()
    query_file_path = f"./data/{unique_id}_{query_file.filename}"
    output_file_result = f"./data/result_{unique_id}.xlsx"

    # 保存上传的文件
    with open(query_file_path, "wb") as f:
        f.write(await query_file.read())
    
    # 提交后台任务
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


@app.get("/download_result/")
async def download_result(unique_id: str):
    status_info = file_status.get(unique_id, None)
    if not status_info or status_info["status"] != "completed":
        return {"error": "File not found or processing not completed."}

    result_file = status_info["result_file"]
    if not os.path.exists(result_file):
        return {"error": "Result file not found."}

    # 设置下载文件的media_type为Excel文件
    return FileResponse(path=result_file, filename=os.path.basename(result_file), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006, reload=True)