import re
from pathlib import Path

from agent.llm import _SUMMARY_PREFIX

_MARKER_RE = re.compile(r"<!-- hakobune:(\w+) -->")


def save_session(messages: list, path: str) -> None:
    """
    messagesをMarkdown形式でファイルに保存する。
    元のシステムプロンプト（ツール定義を含む）は除外する。
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""
        # 元のシステムプロンプト（要約でないもの）はスキップ
        if role == "system" and not content.startswith(_SUMMARY_PREFIX):
            continue
        lines.append(f"<!-- hakobune:{role} -->")
        lines.append(content)
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_session(path: str) -> list:
    """
    Markdown形式のセッションファイルを読み込み、messagesリストを返す。
    存在しない場合は空リストを返す。
    """
    p = Path(path)
    if not p.exists():
        return []

    text = p.read_text(encoding="utf-8")
    # re.split with capturing group: ['', role, content, role, content, ...]
    parts = _MARKER_RE.split(text)

    messages = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip()
        content = parts[i + 1].strip()
        messages.append({"role": role, "content": content})
        i += 2

    return messages
