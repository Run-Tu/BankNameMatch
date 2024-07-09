import faiss
import torch
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def index(model, index_factory="Flat", save_path=None):
    """
    1. 编码整个语料库生成密集嵌入；
    2. 创建Faiss索引；
    3. 可选地保存嵌入。
    """
    test = model.encode("test")
    dtype = test.dtype
    dim = len(test)

    corpus_embeddings = np.memmap(
        save_path,
        mode="r",
        dtype=dtype
    ).reshape(-1, dim)
    
    # 创建Faiss索引
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    if model.device == torch.device("cuda"):
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # Faiss只接受float32类型的数据
    logger.info("Adding embeddings...")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)

    return faiss_index

def search(model, queries, faiss_index, k=100, batch_size=256, max_length=512):
    """
    1. 编码查询生成密集嵌入；
    2. 使用Faiss索引进行检索。

    Args:
        model (FlagModel): 编码模型。
        queries (list): 查询列表。
        faiss_index (faiss.Index): Faiss索引。
        k (int): 检索返回的邻居数量。
        batch_size (int): 批量大小。
        max_length (int): 最大长度。

    Returns:
        tuple: 检索得分和索引。
    """
    query_texts = [query['query'] for query in queries]  # 假设查询是包含'query'键的字典列表
    query_embeddings = model.encode_queries(query_texts, batch_size=batch_size, max_length=max_length)
    query_size = len(query_embeddings)
    
    all_scores = []
    all_indices = []
    
    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    
    return all_scores, all_indices