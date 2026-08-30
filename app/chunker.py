"""切块：读取笔记库，把每篇笔记切成适合检索的块。

策略（适配本笔记库风格）：
1. 遍历 vault_path 下所有 .md，跳过 exclude_dirs 中的目录
2. 按空行拆成段落；段落不超过 chunk_size 直接成块
3. 超长段落按 chunk_size 硬切，相邻块带 chunk_overlap 重叠
4. 产出：块文本 + 出处（相对路径）
"""
from pathlib import Path

from .config import load_config


def iter_markdown_files(vault: Path, exclude_dirs: list[str]) -> list[Path]:
    """返回笔记库中所有 .md 文件（跳过排除目录）。"""
    files = []
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        files.append(p)
    return sorted(files)


def split_paragraphs(text: str) -> list[str]:
    """按空行拆段落，去掉全空段落。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def hard_split(para: str, chunk_size: int, overlap: int) -> list[str]:
    """超长段落按字数硬切，带重叠。"""
    if len(para) <= chunk_size:
        return [para]
    step = chunk_size - overlap
    return [para[i : i + chunk_size] for i in range(0, len(para), step)]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """整篇笔记 → 块列表。"""
    chunks = []
    for para in split_paragraphs(text):
        for piece in hard_split(para, chunk_size, overlap):
            chunks.append(piece)
    return chunks


def build_chunks(cfg: dict) -> list[tuple[Path, list[str]]]:
    """返回 [(相对路径, 块列表), ...]"""
    vault = Path(cfg["vault_path"])
    exclude_dirs = cfg.get("exclude_dirs", [])
    chunk_size = int(cfg.get("chunk_size", 500))
    overlap = int(cfg.get("chunk_overlap", 50))

    if not vault.is_dir():
        raise SystemExit(f"笔记库路径不存在：{vault}")

    result = []
    for p in iter_markdown_files(vault, exclude_dirs):
        text = p.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, chunk_size, overlap)
        result.append((p, chunks))
    return result
