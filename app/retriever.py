"""检索：问题向量化 → Chroma 相似度检索 → top-k + 相似度门槛。

流程：
1. 问题用同一个 BGE-M3 向量化（归一化）
2. Chroma 按余弦相似度检索 top_k 个块
3. 相似度低于门槛的结果丢弃；全部低于门槛判「没找到」
4. 返回：命中块 + 相似度 + 出处（文件、标题链）
"""
from pathlib import Path

from .chunker import CHUNK_VERSION
from .config import load_config
from .indexer import COLLECTION_NAME, _cuda_available, _get_embedder


def _get_retriever_components(cfg: dict, data_dir: Path):
    """返回 (embedder, collection)。embedder 复用建索引时的加载逻辑。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
    collection = client.get_collection(COLLECTION_NAME)
    if (collection.metadata or {}).get("chunk_version") != CHUNK_VERSION:
        raise ValueError("索引仍为旧切块版本，请确认后重建索引再检索。")
    embedder = _get_embedder(data_dir / "models")
    return embedder, collection


def restore_hits(results: dict, threshold: float) -> list[dict]:
    """过门槛的小块恢复完整原始单元；同文件、同单元只返回一次。"""
    grouped = {}
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0],
                               results["distances"][0]):
        if meta.get("chunk_version") != CHUNK_VERSION:
            raise ValueError("索引缺少完整上下文，请确认后重建索引。")
        score = 1 - dist
        if score < threshold:
            continue
        key = (meta["source"], meta["parent_id"])
        if key not in grouped:
            grouped[key] = {
                "file": meta["source"], "title_chain": meta["title_chain"],
                "text": meta["context"], "score": score,
                "parent_id": meta["parent_id"], "matched_chunks": [doc],
            }
        else:
            grouped[key]["score"] = max(grouped[key]["score"], score)
            grouped[key]["matched_chunks"].append(doc)
    hits = sorted(grouped.values(), key=lambda h: h["score"], reverse=True)
    for hit in hits:
        hit["score"] = round(hit["score"], 3)
    return hits


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

    hits = restore_hits(results, threshold)

    return {"query": query, "hits": hits, "found": bool(hits)}
