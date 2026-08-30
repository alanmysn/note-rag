"""配置加载：读取 config.yaml 与 .env。

- config.yaml：笔记库路径、检索参数等（gitignore，本地真实值）
- .env：DeepSeek API key（gitignore）
- 模型缓存目录指向 data/models，避免默认装进 C 盘
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> tuple[dict, Path, Path]:
    """返回 (配置字典, 项目根目录, 数据目录)。缺文件/缺必填项时报错退出。"""
    load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"缺少配置文件：{cfg_path}\n"
            "请复制 config.example.yaml 为 config.yaml 并填写笔记库路径。"
        )

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not cfg.get("vault_path"):
        raise SystemExit("config.yaml 缺少 vault_path（笔记库路径），请填写。")

    data_dir = (PROJECT_ROOT / cfg.get("data_dir", "data")).resolve()
    models_dir = data_dir / "models"
    # 模型缓存指到项目内，避免默认装进 C 盘
    os.environ.setdefault("HF_HOME", str(models_dir))

    return cfg, PROJECT_ROOT, data_dir
