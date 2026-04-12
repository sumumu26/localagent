from dataclasses import dataclass
import argparse

DEFAULT_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a helpful assistant with access to tools.\n\n"
    "Rules:\n"
    "- For ANY question about facts, people, events, games, news, or specific details, "
    "ALWAYS use web_search first. Do NOT answer from memory.\n"
    "- If a follow-up question refers to something discussed earlier, search again to get accurate details.\n"
    "- Only answer without searching for casual conversation or simple math/logic."
)


@dataclass
class Config:
    model_path: str
    chat_format: str = "chatml"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.0
    max_tokens: int = -1
    max_iterations: int = 10
    context_threshold: float = 0.8
    keep_recent: int = 6
    session_file: str = ""
    resume: bool = False
    verbose: bool = False
    verbose_tools: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    settings_path: str = "settings.json"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Local LLM Agent")
    p.add_argument("--model", required=True, help="Path to GGUF model file")
    p.add_argument(
        "--chat-format",
        default="chatml",
        help="llama-cpp chat format (chatml=Qwen3.5推奨, auto=GGUFから検出, llama-3, qwen, ...)",
    )
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="GPU layers to offload (-1 = all)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=-1,
                   help="出力トークン数の上限 (-1 = n_ctx に委ねる, デフォルト: -1)")
    p.add_argument("--max-iterations", type=int, default=10,
                   help="Max ReAct loop iterations before giving up")
    p.add_argument("--context-threshold", type=float, default=0.8,
                   help="コンテキスト圧縮を開始するトークン使用率 (0.0-1.0, デフォルト: 0.8)")
    p.add_argument("--keep-recent", type=int, default=6,
                   help="要約せずに保持する最近のメッセージ数 (デフォルト: 6)")
    p.add_argument("--session", default="",
                   help="セッションファイルのパス (.md)。指定すると再開・自動保存が有効になる")
    p.add_argument("--resume", action="store_true",
                   help="保存済みセッションの一覧を表示して再開するセッションを選択する")
    p.add_argument("--verbose", action="store_true",
                   help="Enable llama-cpp-python verbose output")
    p.add_argument("--verbose-tools", action="store_true",
                   help="ツール呼び出しをパネル形式で詳細表示する（デフォルトは1行表示）")
    p.add_argument("--system-prompt", default=None,
                   help="Override the default system prompt")
    p.add_argument("--settings", default="settings.json",
                   help="Path to settings.json for permission rules (default: settings.json)")
    args = p.parse_args()

    return Config(
        model_path=args.model,
        chat_format=args.chat_format,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        context_threshold=args.context_threshold,
        keep_recent=args.keep_recent,
        session_file=args.session,
        resume=args.resume,
        verbose=args.verbose,
        verbose_tools=args.verbose_tools,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        settings_path=args.settings,
    )
