from llama_cpp import Llama
from config import Config


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
        messages=messages,
        tools=tools if tools else None,
        tool_choice="auto",
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
