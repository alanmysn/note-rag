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
    print(f"  切块大小/重叠 : {cfg.get('chunk_size')} / {cfg.get('chunk_overlap')}")
    print(f"  top-k / 门槛  : {cfg.get('top_k')} / {cfg.get('similarity_threshold')}")
    print(f"  排除文件夹    : {cfg.get('exclude_dirs')}")
    print(f"  DeepSeek key  : {_api_key_status()}")


def cmd_stats() -> None:
    from .chunker import build_chunks, chunk_body_len
    from .config import load_config

    cfg, _, _ = load_config()
    items = build_chunks(cfg)

    total_chunks = sum(len(chunks) for _, chunks in items)
    total_chars = sum(
        len(block) for _, chunks in items for _, block in chunks
    )
    with_chain = sum(
        1 for _, chunks in items for chain, _ in chunks if chain
    )
    big_files = [
        p.name
        for p, chunks in items
        if sum(len(block) for _, block in chunks) > 20000
    ]

    print("note-rag 笔记统计")
    print(f"  笔记篇数      : {len(items)}")
    print(f"  切块总数      : {total_chunks}")
    print(f"  内容总字数    : {total_chars:,}")
    print(f"  平均每篇块数  : {total_chunks / len(items):.1f}")
    print(f"  带标题链的块  : {with_chain} ({with_chain / total_chunks:.0%})")
    print(f"  最长的笔记    : {big_files if big_files else '无'}")
    print(f"  单块最大字数  : {max((chunk_body_len(block) for _, cs in items for _, block in cs), default=0)}")
    print(f"  块大小超 120% 阈值的块数（硬切遗漏检查）: "
          f"{sum(1 for _, cs in items for _, block in cs if chunk_body_len(block) > int(cfg['chunk_size']) * 1.2)}")


def cmd_index(limit: int | None = None) -> None:
    from .indexer import build_index

    stats = build_index(limit=limit)
    print(f"建索引完成：{stats['files']} 篇笔记，{stats['blocks']} 块已入库")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="note-rag")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("config", help="打印当前配置")
    sub.add_parser("stats", help="统计笔记与切块")
    index_p = sub.add_parser("index", help="全量建索引")
    index_p.add_argument("--limit", type=int, default=None,
                         help="只索引前 N 篇（冒烟测试用）")

    args = parser.parse_args()
    if args.cmd in (None, "config"):
        cmd_config()
    elif args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "index":
        cmd_index(args.limit)


if __name__ == "__main__":
    main()
