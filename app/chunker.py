"""切块：读取笔记库，把每篇笔记切成适合检索的块。

策略（2026-08-28 修正：标题感知切分，适配本笔记库实际风格）：
1. 遍历 vault_path 下所有 .md，跳过 exclude_dirs 中的目录
2. 按 Markdown 标题（#/##/###…）优先切分：每个标题下的内容自成一段
3. 无标题的笔记/段落：按空行段落切（兜底）
4. 超长段落按 chunk_size 硬切，相邻块带 chunk_overlap 重叠
5. 每块附带「标题链」上下文（如 `标普500 > 误区`），并记录来源文件
6. 产出：块文本（含标题链前缀）+ 出处（相对路径、标题链）
"""
import re
from pathlib import Path

from .config import load_config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*[。．.!！?？:：]?$")


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


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """按标题切分，返回 [(标题链, 该标题下的内容), ...]。

    标题链用于给块补上下文（如 `定义 > 误区`），无标题内容标题链为空串。
    识别的标题：Markdown # 语法 + 独立成行的加粗（本笔记库常用加粗当小节标题）。
    """
    sections: list[tuple[str, str]] = []
    chain: list[str] = []      # 当前标题链（各级标题文本）
    current: list[str] = []    # 当前标题下的内容行

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            sections.append((" > ".join(chain), body))
        current.clear()

    for line in text.splitlines():
        if line.strip() in ("---", "***", "___"):   # 分隔线不进内容
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            chain = chain[: level - 1] + [title]
            continue
        b = BOLD_HEADING_RE.match(line)
        if b:
            flush()
            chain = [b.group(1).strip()]
            continue
        current.append(line)
    flush()
    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, str]]:
    """整篇笔记 → [(标题链, 块文本), ...]。

    每个标题小节内部再按段落/字数细分；每块文本带上标题链前缀。
    """
    chunks: list[tuple[str, str]] = []
    for chain, body in split_by_headings(text):
        for para in split_paragraphs(body):
            for piece in hard_split(para, chunk_size, overlap):
                prefix = f"[{chain}]\n" if chain else ""
                chunks.append((chain, prefix + piece))
    return chunks


def chunk_body_len(block: str) -> int:
    """块文本去掉标题链前缀后的正文长度（用于超限检查）。"""
    if block.startswith("["):
        nl = block.find("\n")
        if nl != -1:
            return len(block[nl + 1 :])
    return len(block)


def build_chunks(cfg: dict) -> list[tuple[Path, list[tuple[str, str]]]]:
    """返回 [(相对路径, [(标题链, 块文本), ...]), ...]"""
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
