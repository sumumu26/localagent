from dataclasses import dataclass
import argparse

DEFAULT_SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a helpful assistant. "
    "Use tools when you need to look up information, search the web, or read files."
)


@dataclass
class Config:
    model_path: str
    chat_format: str = "auto"
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    temperature: float = 0.0
    max_tokens: int = 1024
    max_iterations: int = 10
    verbose: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Local LLM Agent")
    p.add_argument("--model", required=True, help="Path to GGUF model file")
    p.add_argument(
        "--chat-format",
        default="auto",
        help="llama-cpp chat format (auto=detect from GGUF, or llama-3, chatml, qwen, ...)",
    )
    p.add_argument("--n-ctx", type=int, default=8192)
    p.add_argument("--n-gpu-layers", type=int, default=-1,
                   help="GPU layers to offload (-1 = all)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-iterations", type=int, default=10,
                   help="Max ReAct loop iterations before giving up")
    p.add_argument("--verbose", action="store_true",
                   help="Enable llama-cpp-python verbose output")
    p.add_argument("--system-prompt", default=None,
                   help="Override the default system prompt")
    args = p.parse_args()

    return Config(
        model_path=args.model,
        chat_format=args.chat_format,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
    )
