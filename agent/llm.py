import re
from llama_cpp import Llama
from config import Config

try:
    from rich.console import Console
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False

_SUMMARY_PREFIX = "[これ以前の会話の要約]"


def _fix_surrogates(text: str) -> str:
    """
    サロゲート文字を正しいUnicodeに復元する。
    Python の input() が端末から日本語バイト列を surrogateescape で
    デコードした場合（\udce3 等）に発生するエラーを修正する。
    """
    try:
        return text.encode("utf-8", errors="surrogateescape").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_messages(messages: list) -> list:
    result = []
    for msg in messages:
        sanitized = dict(msg)
        if isinstance(msg.get("content"), str):
            sanitized["content"] = _fix_surrogates(msg["content"])
        result.append(sanitized)
    return result


def load_model(cfg: Config) -> Llama:
    # chat_format=None triggers auto-detection from GGUF metadata (Llama3, Qwen2.5, etc.)
    chat_format = None if cfg.chat_format == "auto" else cfg.chat_format
    return Llama(
        model_path=cfg.model_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        chat_format=chat_format,
        verbose=cfg.verbose,
    )


def _count_tokens(llm: Llama, messages: list) -> int:
    """メッセージリストのトークン数を概算する。"""
    text = "".join(m.get("content", "") or "" for m in messages)
    try:
        return len(llm.tokenize(text.encode("utf-8", errors="replace")))
    except Exception:
        return len(text) // 4


def _summarize_messages(llm: Llama, messages_to_summarize: list, cfg: Config) -> str:
    """古いメッセージをLLMで要約する。"""
    history_text = "\n".join(
        f"{m['role'].upper()}: {m.get('content', '') or ''}"
        for m in messages_to_summarize
    )
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a summarization assistant. "
                "Summarize the following conversation history concisely, "
                "preserving all important facts, decisions, code changes, file paths, findings, and context. "
                "If the conversation is in Japanese, summarize in Japanese."
            ),
        },
        {
            "role": "user",
            "content": f"以下の会話履歴を要約してください:\n\n{history_text}",
        },
    ]
    response = llm.create_chat_completion(
        messages=_sanitize_messages(prompt),
        max_tokens=512,
        temperature=0.0,
    )
    content = response["choices"][0]["message"].get("content", "") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|channel>thought.*?<channel\|>", "", content, flags=re.DOTALL)
    return content.strip()


def maybe_compress_context(llm: Llama, messages: list, cfg: Config) -> None:
    """
    コンテキストが閾値を超えた場合、古いメッセージを要約して圧縮する。
    messages をin-placeで変更する。
    """
    budget = int(cfg.n_ctx * cfg.context_threshold)
    if _count_tokens(llm, messages) <= budget:
        return

    original_system = [m for m in messages if m["role"] == "system" and not m["content"].startswith(_SUMMARY_PREFIX)]
    existing_summary = [m for m in messages if m["role"] == "system" and m["content"].startswith(_SUMMARY_PREFIX)]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= cfg.keep_recent:
        return  # これ以上圧縮できない

    to_summarize = existing_summary + non_system[:-cfg.keep_recent]
    to_keep = non_system[-cfg.keep_recent:]

    if _USE_RICH:
        _console.print("[yellow][Context] コンテキストが上限に近づきました。過去の会話を要約しています...[/yellow]")
    else:
        print("[Context] コンテキストが上限に近づきました。過去の会話を要約しています...")

    summary_text = _summarize_messages(llm, to_summarize, cfg)

    messages.clear()
    messages.extend(original_system)
    messages.append({"role": "system", "content": f"{_SUMMARY_PREFIX}\n{summary_text}"})
    messages.extend(to_keep)

    if _USE_RICH:
        _console.print("[yellow][Context] 要約完了。会話を継続します。[/yellow]")
    else:
        print("[Context] 要約完了。会話を継続します。")


_MAX_CONTINUATIONS = 10


def chat_completion(llm: Llama, messages: list, tools: list, cfg: Config) -> dict:
    response = llm.create_chat_completion(
        messages=_sanitize_messages(messages),
        tools=tools if tools else None,
        tool_choice="auto",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        partial = choice["message"].get("content") or ""
        full_content = _continue_truncated(llm, messages, partial, cfg)
        response["choices"][0]["message"]["content"] = full_content
        response["choices"][0]["finish_reason"] = "stop"

    return response


def _continue_truncated(llm: Llama, original_messages: list, partial: str, cfg: Config) -> str:
    """
    finish_reason == 'length' で打ち切られた応答を継続する。
    打ち切られた内容をアシスタントメッセージとして渡し、続きを生成させる。
    """
    content = partial
    for i in range(_MAX_CONTINUATIONS):
        if _USE_RICH:
            _console.print(f"[yellow][Continue] 生成が途中で打ち切られました。継続中... ({i + 1}/{_MAX_CONTINUATIONS})[/yellow]")
        else:
            print(f"[Continue] 生成が途中で打ち切られました。継続中... ({i + 1}/{_MAX_CONTINUATIONS})")

        cont_messages = list(original_messages) + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Continue your response from exactly where it was cut off. Output only the continuation, no repetition."},
        ]
        response = llm.create_chat_completion(
            messages=_sanitize_messages(cont_messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        choice = response["choices"][0]
        content += choice["message"].get("content") or ""
        if choice.get("finish_reason") != "length":
            break

    return content
