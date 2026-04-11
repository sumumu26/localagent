import re
from datetime import datetime
from pathlib import Path

from agent.llm import _SUMMARY_PREFIX

_MARKER_RE = re.compile(r"<!-- hakobune:(\w+) -->")
SESSIONS_DIR = "sessions"


def save_session(messages: list, path: str) -> None:
    """
    messagesをMarkdown形式でファイルに保存する。
    元のシステムプロンプト（ツール定義を含む）は除外する。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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


def new_session_path(sessions_dir: str = SESSIONS_DIR) -> str:
    """タイムスタンプ付きの新規セッションファイルパスを返す。"""
    Path(sessions_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(sessions_dir) / f"{timestamp}.md")


def list_sessions(sessions_dir: str = SESSIONS_DIR) -> list:
    """sessions_dir内のセッションファイルを更新日時の新しい順で返す。"""
    d = Path(sessions_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)


def get_latest_user_message(path: str, max_chars: int = 80) -> str:
    """セッションファイルから最新のユーザーメッセージを取得し、truncateして返す。"""
    messages = load_session(path)
    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg["content"].replace("\n", " ").strip()
            if len(content) > max_chars:
                return content[:max_chars] + "..."
            return content
    return "(メッセージなし)"
