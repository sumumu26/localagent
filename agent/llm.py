import re
import llama_cpp
from llama_cpp import Llama
from config import Config

try:
    from rich.console import Console
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False

_SUMMARY_PREFIX = "[これ以前の会話の要約]"

# ---------------------------------------------------------------------------
# KV cache quantization type resolution
# ---------------------------------------------------------------------------
# 標準 llama-cpp-python は GGML_TYPE_* 整数定数を llama_cpp モジュールから公開する。
# TurboQuant 型 (TBQ3_0 / TBQ4_0) はコミュニティ fork が追加:
#   https://github.com/TheTom/llama-cpp-turboquant
# fork からのビルド手順:
#   git clone https://github.com/TheTom/llama-cpp-turboquant
#   cd llama-cpp-turboquant
#   pip install -e . --no-build-isolation \
#       -C cmake.args="-DGGML_CUDA=ON"   # Apple Silicon は -DGGML_METAL=ON
# TBQ4_0: 3.94x 圧縮, 16GB VRAM + Qwen3.5-9B で ~335K context
# TBQ3_0: 5.22x 圧縮, 16GB VRAM + Qwen3.5-9B で ~435K context

_KV_TYPE_MAP: dict = {
    "f32":  0,
    "f16":  1,
    "q4_0": 2,
    "q4_1": 3,
    "q5_0": 6,
    "q5_1": 7,
    "q8_0": 8,
    "bf16": 30,
}

# TurboQuant 型は fork ビルドのみ利用可能 — 動的プローブで検出
# 整数値をハードコードしない（fork によって値が異なる可能性があるため）
_tbq4 = getattr(llama_cpp, "GGML_TYPE_TBQ4_0", None)
_tbq3 = getattr(llama_cpp, "GGML_TYPE_TBQ3_0", None)
if _tbq4 is not None:
    _KV_TYPE_MAP["tbq4_0"] = int(_tbq4)
if _tbq3 is not None:
    _KV_TYPE_MAP["tbq3_0"] = int(_tbq3)


def _resolve_kv_type(name: str, param_name: str) -> int:
    """KV キャッシュ型名を GGML type 整数に解決する。

    利用不可能な型が要求された場合は警告を出して f16 にフォールバックする。
    """
    key = name.lower()
    if key in _KV_TYPE_MAP:
        return _KV_TYPE_MAP[key]
    turboquant_types = {"tbq3_0", "tbq4_0"}
    if key in turboquant_types:
        if _USE_RICH:
            _console.print(
                f"[yellow][KV Cache] {param_name}={name!r} は TurboQuant fork ビルドが必要です "
                f"(https://github.com/TheTom/llama-cpp-turboquant)。f16 で続行します。[/yellow]"
            )
        else:
            print(f"[KV Cache] {param_name}={name!r} は TurboQuant fork ビルドが必要です。f16 で続行します。")
    else:
        if _USE_RICH:
            _console.print(
                f"[yellow][KV Cache] 不明な {param_name}={name!r}。"
                f"有効な型: {', '.join(sorted(_KV_TYPE_MAP))}。f16 で続行します。[/yellow]"
            )
        else:
            print(f"[KV Cache] 不明な {param_name}={name!r}。f16 で続行します。")
    return _KV_TYPE_MAP["f16"]


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
    type_k = _resolve_kv_type(cfg.kv_cache_type_k, "kv_cache_type_k")
    type_v = _resolve_kv_type(cfg.kv_cache_type_v, "kv_cache_type_v")
    return Llama(
        model_path=cfg.model_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        chat_format=chat_format,
        verbose=cfg.verbose,
        type_k=type_k,
        type_v=type_v,
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


def chat_completion(llm: Llama, messages: list, tools: list, cfg: Config) -> dict:
    return llm.create_chat_completion(
        messages=_sanitize_messages(messages),
        tools=tools if tools else None,
        tool_choice="auto",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
