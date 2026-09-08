"""结构切块：小块检索，完整段落/列表用于回答（2026-09-08）。"""
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_VERSION = 3
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*[。．.!！?？:：]?$")
LIST_RE = re.compile(r"^(\s*)(?:[-+*]|\d+[.)])\s+")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass
class Chunk:
    title_chain: str
    body: str
    parent_id: str
    context: str
    file_name: str

    @property
    def text(self) -> str:
        background = " > ".join(p for p in (self.file_name, self.title_chain) if p)
        return f"[{background}]\n{self.body}" if background else self.body


def iter_markdown_files(vault: Path, exclude_dirs: list[str]) -> list[Path]:
    return sorted(p for p in vault.rglob("*.md")
                  if not any(part in exclude_dirs for part in p.relative_to(vault).parts))


def split_by_headings(text: str) -> list[tuple[str, str]]:
    sections = []
    headings: list[tuple[int, str]] = []
    bold = ""
    current: list[str] = []
    fence = ""

    def flush() -> None:
        body = "\n".join(current).strip()
        if body:
            chain = [title for _, title in headings] + ([bold] if bold else [])
            sections.append((" > ".join(chain), body))
        current.clear()

    lines = text.lstrip("\ufeff").splitlines()
    if lines and lines[0].strip() == "---":
        closing = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if closing is None:
            raise ValueError("文件开头的属性区缺少结束标记")
        lines = lines[closing + 1:]
    for line in lines:
        f = FENCE_RE.match(line)
        if fence:
            current.append(line)
            if line.strip().startswith(fence) and set(line.strip()) == {fence[0]}:
                fence = ""
            continue
        if f:
            fence = f.group(1)
            current.append(line)
            continue
        if line.strip() in ("---", "***", "___"):
            flush()
            continue
        m = HEADING_RE.match(line)
        b = BOLD_HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            headings = [(n, title) for n, title in headings if n < level]
            headings.append((level, m.group(2).strip()))
            bold = ""
        elif b:
            flush()
            bold = b.group(1).strip()
        else:
            current.append(line)
    flush()
    return sections


@dataclass
class Unit:
    kind: str
    text: str
    start: int
    end: int
    items: list[str]
    intro: str = ""


def table_start(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines) or "|" not in lines[i]:
        return False
    cells = lines[i + 1].strip().strip("|").split("|")
    return len(cells) > 1 and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in cells)


def ordered_number(item: str):
    m = re.match(r"^(\s*)(\d+)[.)]\s+", item)
    return (len(m.group(1).expandtabs(4)), int(m.group(2))) if m else None


def split_units(body: str, target: int = 500) -> list[Unit]:
    """先识别段落、列表、表格，再关联连续步骤和紧邻说明。"""
    lines = body.splitlines(keepends=True)
    units = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        start = i
        match = LIST_RE.match(lines[i])
        items = []
        if table_start(lines, i):
            kind = "table"
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                i += 1
            items = [line.rstrip("\r\n") for line in lines[start:i]]
        elif match:
            kind = "list"
            indent = len(match.group(1).expandtabs(4))
            starts = [i]
            i += 1
            while i < len(lines):
                if table_start(lines, i):
                    break
                m = LIST_RE.match(lines[i])
                if m:
                    depth = len(m.group(1).expandtabs(4))
                    if depth < indent:
                        break
                    if depth == indent:
                        starts.append(i)
                elif not lines[i].strip():
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j == len(lines):
                        break
                    next_indent = len(lines[j].expandtabs(4)) - len(lines[j].expandtabs(4).lstrip())
                    if table_start(lines, j) or (not LIST_RE.match(lines[j]) and next_indent <= indent):
                        break
                i += 1
            items = ["".join(lines[a:b]).rstrip() for a, b in zip(starts, starts[1:] + [i])]
        else:
            kind = "paragraph"
            fence = FENCE_RE.match(lines[i])
            i += 1
            if fence:
                kind = "code"
                marker = fence.group(1)
                while i < len(lines):
                    line = lines[i].strip()
                    i += 1
                    if line.startswith(marker) and set(line) == {marker[0]}:
                        break
            else:
                while (i < len(lines) and lines[i].strip()
                       and not LIST_RE.match(lines[i]) and not table_start(lines, i)):
                    i += 1
        units.append(Unit(kind, "".join(lines[start:i]).strip(), start, i, items))

    merged = []
    i = 0
    while i < len(units):
        unit = units[i]
        while (unit.kind == "list" and i + 2 < len(units)
               and units[i + 1].kind == "table" and units[i + 2].kind == "list"):
            following = units[i + 2]
            last = ordered_number(unit.items[-1])
            first = ordered_number(following.items[0])
            if last is None or first != (last[0], last[1] + 1):
                break
            unit.items += [units[i + 1].text] + following.items
            unit.end = following.end
            unit.text = "".join(lines[unit.start:unit.end]).strip()
            i += 2
        if (unit.kind in ("list", "table") and merged and merged[-1].kind == "paragraph"
                and (len(merged[-1].text) <= target or merged[-1].text.endswith(("：", ":")))):
            previous = merged.pop()
            unit.intro = previous.text
            unit.start = previous.start
            unit.text = "".join(lines[unit.start:unit.end]).strip()
        merged.append(unit)
        i += 1
    return merged


def split_table(intro: str, rows: list[str], target: int, maximum: int) -> list[str]:
    """按完整数据行切；单行及表头允许超字符上限，模型容量另查。"""
    pieces = []
    prefix = f"{intro}\n\n" if intro else ""
    if len(prefix) >= target:
        pieces.extend(split_long(intro, target, maximum))
        prefix = ""
    header = prefix + "\n".join(rows[:2])
    current = header
    for row in rows[2:]:
        candidate = current + "\n" + row
        if current != header and len(candidate) > target:
            pieces.append(current)
            current = header
        current += "\n" + row
    if current != header or not pieces:
        pieces.append(current)
    return pieces


def split_long(text: str, target: int, maximum: int) -> list[str]:
    """句末优先、句内标点其次、硬切兜底；拼接保持原文。"""
    pieces = []
    while len(text) > maximum:
        window = text[:maximum]
        endings = [m.end() for m in re.finditer(r"[。！？!?]|\.(?=\s)", window)]
        if not endings:
            endings = [m.end() for m in re.finditer(r"[；;，,：:]|\n", window)]
        cut = min(endings, key=lambda n: abs(n - target)) if endings else maximum
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


def split_list(intro: str, items: list[str], target: int, maximum: int) -> list[str]:
    prefix = f"{intro}\n\n" if intro else ""
    pieces = []
    if len(prefix) >= target:
        pieces.extend(split_long(intro, target, maximum))
        prefix = ""
    current = prefix
    for item in items:
        rows = item.splitlines()
        if table_start(rows, 0):
            if current != prefix:
                pieces.append(current)
                current = prefix
            pieces.extend(split_table(intro if prefix else "", rows, target, maximum))
            continue
        separator = "\n" if current != prefix else ""
        if current != prefix and len(current + separator + item) > target:
            pieces.append(current)
            current = prefix
            separator = ""
        if len(prefix + item) > maximum:
            pieces.extend(prefix + part for part in split_long(
                item, max(1, target - len(prefix)), maximum - len(prefix)))
            current = prefix
        else:
            current += separator + item
    if current != prefix:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int = 500, chunk_max_size: int = 800,
               file_name: str = "") -> list[Chunk]:
    if not 0 < chunk_size <= chunk_max_size:
        raise ValueError("必须满足 0 < chunk_size <= chunk_max_size")
    chunks = []
    parent = 0
    for chain, body in split_by_headings(text):
        pending = ""

        def emit(context: str, pieces: list[str]) -> None:
            nonlocal parent
            chunks.extend(Chunk(chain, piece, str(parent), context, file_name) for piece in pieces)
            parent += 1

        for unit in split_units(body, chunk_size):
            original, intro, items = unit.text, unit.intro, unit.items
            if items or len(original) > chunk_size:
                if pending:
                    emit(pending, [pending])
                    pending = ""
                if len(original) <= chunk_max_size:
                    pieces = [original]
                elif unit.kind == "table":
                    pieces = split_table(intro, items, chunk_size, chunk_max_size)
                elif items:
                    pieces = split_list(intro, items, chunk_size, chunk_max_size)
                else:
                    pieces = split_long(original, chunk_size, chunk_max_size)
                emit(original, pieces)
            else:
                candidate = f"{pending}\n\n{original}" if pending else original
                if len(candidate) > chunk_size:
                    emit(pending, [pending])
                    pending = original
                else:
                    pending = candidate
        if pending:
            emit(pending, [pending])
    return chunks


def build_chunks(cfg: dict) -> list[tuple[Path, list[Chunk]]]:
    vault = Path(cfg["vault_path"])
    if not vault.is_dir():
        raise SystemExit(f"笔记库路径不存在：{vault}")
    result = []
    for path in iter_markdown_files(vault, cfg.get("exclude_dirs", [".obsidian"])):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, int(cfg.get("chunk_size", 500)),
                            int(cfg.get("chunk_max_size", 800)), path.name)
        result.append((path.relative_to(vault), chunks))
    return result
