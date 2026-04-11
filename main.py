import sys
import os
from config import parse_args
from agent.llm import load_model
from agent.loop import run_loop
from agent import registry, permissions
from agent.tool_calling import get_adapter
from datetime import datetime
from agent.session import (
    load_session, save_session, new_session_path,
    list_sessions, get_latest_user_message, set_current_session,
)
import agent.tools  # noqa: F401 — triggers all @register decorators

try:
    from rich.console import Console
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False

try:
    from prompt_toolkit import prompt as pt_prompt
    _USE_PROMPT_TOOLKIT = True
except ImportError:
    _USE_PROMPT_TOOLKIT = False


def _ask(prompt_str: str) -> str:
    """Read a line from the user with proper wide-character (CJK) support."""
    if _USE_PROMPT_TOOLKIT:
        return pt_prompt(prompt_str)
    if _USE_RICH:
        from rich.prompt import Prompt
        return Prompt.ask(f"[bold blue]{prompt_str.rstrip()}[/bold blue]")
    return input(prompt_str)


def _configure_windows_encoding() -> None:
    """Windows環境でUTF-8入出力を有効にする。"""
    if sys.platform != "win32":
        return
    # Python 3.15未満ではUTF-8モードが自動でないため明示的に設定
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    # cmd.exe のコードページをUTF-8に切り替え
    os.system("chcp 65001 >nul 2>&1")


def _pick_session() -> str | None:
    """保存済みセッションを一覧表示し、選択されたパスを返す。キャンセル時はNone。"""
    sessions = list_sessions()
    if not sessions:
        if _USE_RICH:
            _console.print("[yellow]保存済みセッションがありません。新規セッションを開始します。[/yellow]\n")
        else:
            print("保存済みセッションがありません。新規セッションを開始します。\n")
        return None

    if _USE_RICH:
        _console.print("[bold]利用可能なセッション:[/bold]\n")
    else:
        print("利用可能なセッション:\n")

    for i, path in enumerate(sessions, 1):
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        preview = get_latest_user_message(str(path))
        if _USE_RICH:
            _console.print(f"  [cyan][{i}][/cyan] {mtime}  {preview}")
        else:
            print(f"  [{i}] {mtime}  {preview}")

    print()
    try:
        if _USE_RICH:
            choice = _console.input("[bold]番号を選択[/bold] (Enter で新規セッション開始): ").strip()
        else:
            choice = input("番号を選択 (Enter で新規セッション開始): ").strip()
        if not choice:
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return str(sessions[idx])
    except (ValueError, KeyboardInterrupt):
        pass
    return None


def main() -> None:
    _configure_windows_encoding()
    cfg = parse_args()
    permissions.load(cfg.settings_path)

    if _USE_RICH:
        _console.print(f"[bold]Loading model:[/bold] {cfg.model_path}")
    else:
        print(f"Loading model: {cfg.model_path}")

    llm = load_model(cfg)

    if _USE_RICH:
        _console.print("[bold green]Model loaded.[/bold green] Type [bold]exit[/bold] or Ctrl-C to quit.\n")
    else:
        print("Model loaded. Type 'exit' or Ctrl-C to quit.\n")

    # Embed tool definitions into the system prompt using the model-specific adapter
    adapter = get_adapter(cfg.chat_format)
    system_content = adapter.build_system_prompt(
        cfg.system_prompt, registry.get_tool_definitions()
    )
    messages = [{"role": "system", "content": system_content}]

    # セッションファイルパスを決定
    if cfg.resume:
        session_path = _pick_session() or new_session_path()
    elif cfg.session_file:
        session_path = cfg.session_file
    else:
        session_path = new_session_path()

    set_current_session(session_path)

    # history = 圧縮なしの完全な会話履歴（セッションファイルと同期）
    history = load_session(session_path)
    if history:
        messages.extend(history)
        if _USE_RICH:
            _console.print(f"[bold green]Session restored:[/bold green] {session_path} ({len(history)} messages)\n")
        else:
            print(f"Session restored: {session_path} ({len(history)} messages)\n")
    else:
        if _USE_RICH:
            _console.print(f"[dim]Session: {session_path}[/dim]\n")
        else:
            print(f"Session: {session_path}\n")

    try:
        while True:
            try:
                user_input = _ask("You: ").strip()
            except EOFError:
                break

            if user_input.lower() in ("exit", "quit", "q"):
                break
            if not user_input:
                continue

            user_msg = {"role": "user", "content": user_input}
            messages.append(user_msg)
            history.append(user_msg)

            before_count = len(messages)
            response = run_loop(llm, messages, cfg)
            # run_loop がmessagesに追加した分（assistant応答・ツール結果）をhistoryに同期
            history.extend(messages[before_count:])

            if _USE_RICH:
                _console.print(f"\n[bold green]Assistant:[/bold green] {response}\n")
            else:
                print(f"\nAssistant: {response}\n")

            save_session(history, session_path)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        del llm


if __name__ == "__main__":
    main()
