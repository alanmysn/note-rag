"""note-rag 命令行入口。

用法：
    python -m app.main config     # 打印当前配置
"""
import os


def _api_key_status() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    return "已设置" if key else "未设置（第 5 步前填入 .env）"


def cmd_config() -> None:
    from .config import load_config

    cfg, _, data_dir = load_config()
    print("note-rag 配置")
    print(f"  笔记库路径    : {cfg['vault_path']}")
    print(f"  数据目录      : {data_dir}")
    print(f"  向量化来源    : {cfg.get('embedding_provider', 'local-bge-m3')}")
    print(f"  合并目标/上限 : {cfg.get('chunk_size', 500)} / {cfg.get('chunk_max_size', 800)}")
    print(f"  top-k / 门槛  : {cfg.get('top_k')} / {cfg.get('similarity_threshold')}")
    print(f"  排除文件夹    : {cfg.get('exclude_dirs')}")
    print(f"  DeepSeek key  : {_api_key_status()}")


def cmd_stats() -> None:
    from .chunker import build_chunks
    from .config import load_config

    cfg, _, _ = load_config()
    items = build_chunks(cfg)

    total_chunks = sum(len(chunks) for _, chunks in items)
    total_chars = sum(
        len(chunk.body) for _, chunks in items for chunk in chunks
    )
    with_chain = sum(
        1 for _, chunks in items for chunk in chunks if chunk.title_chain
    )
    big_files = [
        p.name
        for p, chunks in items
        if sum(len(chunk.body) for chunk in chunks) > 20000
    ]

    print("note-rag 笔记统计")
    print(f"  笔记篇数      : {len(items)}")
    print(f"  切块总数      : {total_chunks}")
    print(f"  内容总字数    : {total_chars:,}")
    print(f"  平均每篇块数  : {total_chunks / len(items) if items else 0:.1f}")
    print(f"  带标题链的块  : {with_chain} ({with_chain / total_chunks if total_chunks else 0:.0%})")
    print(f"  最长的笔记    : {big_files if big_files else '无'}")
    print(f"  单块最大字数  : {max((len(chunk.body) for _, cs in items for chunk in cs), default=0)}")
    print(f"  正文超过上限的块数: "
          f"{sum(1 for _, cs in items for chunk in cs if len(chunk.body) > int(cfg.get('chunk_max_size', 800)))}")


def cmd_index(limit: int | None = None) -> None:
    from .indexer import build_index

    stats = build_index(limit=limit)
    print(f"建索引完成：{stats['files']} 篇笔记，{stats['blocks']} 块已入库")


def cmd_search(query: str, limit: int | None = None) -> None:
    from .retriever import search

    result = search(query, limit=limit)
    print(f"问题：{result['query']}")
    if not result["found"]:
        print("未命中：全库检索低于相似度门槛，判「没找到」")
        return
    print(f"命中 {len(result['hits'])} 块：")
    for h in result["hits"]:
        print(f"\n  [{h['score']:.2f}] {h['file']}")
        if h["title_chain"]:
            print(f"      标题链: {h['title_chain']}")
        print(f"      {h['text'][:80]}…")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="note-rag")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("config", help="打印当前配置")
    sub.add_parser("stats", help="统计笔记与切块")
    index_p = sub.add_parser("index", help="全量建索引")
    index_p.add_argument("--limit", type=int, default=None,
                         help="只索引前 N 篇（冒烟测试用）")
    search_p = sub.add_parser("search", help="检索（不生成答案）")
    search_p.add_argument("query", help="要检索的问题")
    search_p.add_argument("--limit", type=int, default=None,
                          help="覆盖 top_k 配置")

    args = parser.parse_args()
    if args.cmd in (None, "config"):
        cmd_config()
    elif args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "index":
        cmd_index(args.limit)
    elif args.cmd == "search":
        cmd_search(args.query, args.limit)


if __name__ == "__main__":
    main()
