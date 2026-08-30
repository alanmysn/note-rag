"""检索：问题向量化 → Chroma 相似度检索 → top-k + 相似度门槛。

流程：
1. 问题用同一个 BGE-M3 向量化（归一化）
2. Chroma 按余弦相似度检索 top_k 个块
3. 相似度低于门槛的结果丢弃；全部低于门槛判「没找到」
4. 返回：命中块 + 相似度 + 出处（文件、标题链）
"""
from pathlib import Path

from .config import load_config
from .indexer import COLLECTION_NAME, _cuda_available, _get_embedder


def _get_retriever_components(cfg: dict, data_dir: Path):
    """返回 (embedder, collection)。embedder 复用建索引时的加载逻辑。"""
    embedder = _get_embedder(data_dir / "models")
    import chromadb

    client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    return embedder, collection


def search(query: str, limit: int | None = None) -> dict:
    """检索。返回：
    {
        "query": 问题,
        "hits": [ {file, title_chain, text, score}, ... ],
        "found": bool,
    }
    """
    cfg, _, data_dir = load_config()
    top_k = int(limit or cfg.get("top_k", 5))
    threshold = float(cfg.get("similarity_threshold", 0.5))

    embedder, collection = _get_retriever_components(cfg, data_dir)

    vec = embedder.encode([query], normalize_embeddings=True)[0]

    results = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist  # Chroma 余弦距离 → 相似度
        if score < threshold:
            continue
        hits.append(
            {
                "file": meta["source"],
                "title_chain": meta.get("title_chain", ""),
                "text": doc,
                "score": round(score, 3),
            }
        )

    return {"query": query, "hits": hits, "found": bool(hits)}
