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


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "config":
        cmd_config()
    else:
        print(f"未知命令：{args[0]}（当前支持：config）")
        sys.exit(1)


if __name__ == "__main__":
    main()
