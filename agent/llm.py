from llama_cpp import Llama
from config import Config


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


def chat_completion(llm: Llama, messages: list, tools: list, cfg: Config) -> dict:
    return llm.create_chat_completion(
        messages=_sanitize_messages(messages),
        tools=tools if tools else None,
        tool_choice="auto",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
