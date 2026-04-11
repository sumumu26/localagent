import re
import subprocess
import sys
from agent.registry import register

_TIMEOUT_SECONDS = 30
_MAX_OUTPUT_CHARS = 8192
_IS_WINDOWS = sys.platform == "win32"

# 実行を完全に禁止するパターン（Linux/Mac/WSL および Windows）
_BLOCKED_PATTERNS = [
    # Linux / Mac / WSL / Git Bash
    (r"\brm\b[^\n]*-[a-zA-Z]*r", "rm -r（再帰削除）は禁止されています"),
    (r"\brm\b[^\n]*--recursive", "rm --recursive は禁止されています"),
    (r"\bdd\b[^\n]*\bof=/dev/", "dd によるデバイス書き込みは禁止されています"),
    (r"\bmkfs\b", "mkfs（ファイルシステムフォーマット）は禁止されています"),
    # Windows (cmd.exe / PowerShell)
    (r"\brd\b[^\n]*/[sS]", "rd /s（再帰削除）は禁止されています"),
    (r"\brmdir\b[^\n]*/[sS]", "rmdir /s（再帰削除）は禁止されています"),
    (r"\bdel\b[^\n]*/[sS]", "del /s（再帰削除）は禁止されています"),
    (r"\bformat\b\s+[a-zA-Z]:", "format（ドライブフォーマット）は禁止されています"),
    (r"\bRemove-Item\b[^\n]*-Recurse", "Remove-Item -Recurse（再帰削除）は禁止されています"),
]

try:
    from rich.console import Console
    from rich.prompt import Confirm
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False


def _blocked_reason(command: str) -> str | None:
    """禁止コマンドならその理由を返す。問題なければNone。"""
    for pattern, reason in _BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return reason
    return None


def _confirm(command: str) -> bool:
    """ユーザーに実行確認を求める。デフォルトはNo。"""
    if _USE_RICH:
        _console.print(f"\n[yellow]実行するコマンド:[/yellow] {command}")
        return Confirm.ask("実行しますか？", default=False)
    else:
        print(f"\n実行するコマンド: {command}")
        answer = input("実行しますか？ [y/N]: ").strip().lower()
        return answer == "y"


@register({
    "name": "shell",
    "description": (
        "シェルコマンドを実行し、標準出力と標準エラーを返す。"
        "git, grep, pytest, make, curl などの外部コマンドを実行できる。"
        f"タイムアウトは {_TIMEOUT_SECONDS} 秒。"
        "Windows では cmd.exe、Linux/Mac では /bin/sh を使用。"
        "rm -r / rd /s など再帰削除・フォーマット系コマンドは禁止。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "実行するシェルコマンド",
            },
        },
        "required": ["command"],
    },
})
def shell(command: str) -> str:
    reason = _blocked_reason(command)
    if reason:
        return f"[Error] コマンドがブロックされました: {reason}"

    if not _confirm(command):
        return "[Cancelled] ユーザーがコマンドの実行をキャンセルしました。"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        output = result.stdout + result.stderr
        if not output:
            output = f"(exit code: {result.returncode})"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n... (出力が長いため {_MAX_OUTPUT_CHARS} 文字でカット)"
        return output
    except subprocess.TimeoutExpired:
        return f"[Error] {_TIMEOUT_SECONDS} 秒でタイムアウトしました。"
    except Exception as e:
        return f"[Error] {e}"
