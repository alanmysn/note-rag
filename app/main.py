"""note-rag 命令行入口。

用法：
    python -m app.main config     # 打印当前配置
"""
import os
import sys


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


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "config":
        cmd_config()
    elif args[0] == "stats":
        cmd_stats()
    else:
        print(f"未知命令：{args[0]}（当前支持：config, stats）")
        sys.exit(1)


if __name__ == "__main__":
    main()
