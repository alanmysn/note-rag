"""向量化与建索引：BGE-M3 本地模型 + Chroma 存储。

流程：
1. 加载本地 BGE-M3（首次自动下载到 data/models，1-2 GB）
2. 对每个块向量化（GPU 可用则自动用 GPU）
3. 连同原文、出处（相对路径、标题链、块序号）存入 Chroma
4. 幂等：建索引前清空旧索引，重跑不产生重复块
"""
from pathlib import Path

from .chunker import build_chunks
from .config import load_config

COLLECTION_NAME = "notes"


def _get_embedder(model_dir: Path):
    """加载 BGE-M3 模型。

    优先加载本地 data/models/bge-m3（魔搭下载）；没有则从 HF 下载。
    """
    from sentence_transformers import SentenceTransformer

    local_model = model_dir / "bge-m3"
    if local_model.is_dir() and any(local_model.iterdir()):
        print(f"使用本地模型：{local_model}")
        return SentenceTransformer(
            str(local_model),
            device="cuda" if _cuda_available() else "cpu",
        )

    print("本地无模型，从 Hugging Face 下载…")
    return SentenceTransformer(
        "BAAI/bge-m3",
        cache_folder=str(model_dir),
        device="cuda" if _cuda_available() else "cpu",
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _get_collection(data_dir: Path):
    """打开 Chroma 集合（首次自动创建）。"""
    import chromadb

    client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 余弦相似度（看方向）
    )


def build_index(limit: int | None = None) -> dict:
    """全量建索引。limit 用于冒烟测试（只索引前 N 篇）。

    返回统计信息。
    """
    from chromadb.api.types import Documents, Embeddings, IDs, Metadatas

    cfg, _, data_dir = load_config()
    items = build_chunks(cfg)
    if limit:
        items = items[:limit]

    print(f"加载 BGE-M3 模型（首次运行会下载，约 1-2 GB）…")
    embedder = _get_embedder(data_dir / "models")

    collection = _get_collection(data_dir)
    # 幂等：清空后重建
    ids_in_collection = collection.get()["ids"]
    if ids_in_collection:
        collection.delete(ids_in_collection)

    total_blocks = sum(len(chunks) for _, chunks in items)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    n = 0
    for rel_path, chunks in items:
        for idx, (chain, block) in enumerate(chunks):
            file_key = rel_path.as_posix()
            ids.append(f"{file_key}#{idx}")
            docs.append(block)
            metas.append(
                {
                    "source": file_key,
                    "title_chain": chain,
                    "chunk_index": idx,
                }
            )
            n += 1

    print(f"向量化 {n} 块（共 {len(items)} 篇笔记）…")
    embeddings: list[list[float]] = []
    batch = 64
    for i in range(0, len(docs), batch):
        batch_docs = docs[i : i + batch]
        vecs = embedder.encode(batch_docs, normalize_embeddings=True)
        embeddings.extend(vecs.tolist())
        if (i // batch + 1) % 5 == 0 or i + batch >= len(docs):
            print(f"  已处理 {min(i + batch, len(docs))}/{len(docs)} 块")

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas,
    )

    return {"files": len(items), "blocks": n}
