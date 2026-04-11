import json
import re
from agent import registry
from agent.tool_calling import get_adapter
from agent.llm import chat_completion, maybe_compress_context
from config import Config

try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False


def run_loop(llm, messages: list, cfg: Config) -> str:
    """
    Runs the ReAct loop. Mutates `messages` in place.
    Returns the final text response.
    """
    adapter = get_adapter(cfg.chat_format)

    for iteration in range(cfg.max_iterations):
        maybe_compress_context(llm, messages, cfg)
        # tools=None: tool definitions are already embedded in the system prompt
        response = chat_completion(llm, messages, tools=None, cfg=cfg)

        msg = response["choices"][0]["message"]
        content = _strip_thinking(msg.get("content") or "")

        messages.append({"role": "assistant", "content": content})

        calls = adapter.extract_tool_calls(content)

        if not calls:
            # No tool calls → final answer
            return adapter.strip_tool_calls(content)

        # Execute all tool calls and collect results
        results = []
        for tc in calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            args_json = json.dumps(args, ensure_ascii=False)

            _print_tool_call(name, args_json, cfg.verbose_tools)
            result = registry.dispatch(name, args_json)
            _print_tool_result(name, result, cfg.verbose_tools)

            results.append(adapter.format_tool_result(name, result))

        messages.append({
            "role": "user",
            "content": "\n".join(results),
        })

    return (
        f"[Agent stopped after {cfg.max_iterations} iterations. "
        "The last tool results are in the conversation history.]"
    )


def _strip_thinking(text: str) -> str:
    """モデルごとの thinking ブロックを除去する。"""
    # Qwen3.5: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Gemma 4: <|channel>thought ... <channel|>
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


def _print_tool_call(name: str, args_json: str, verbose: bool = False) -> None:
    try:
        args = json.loads(args_json)
        args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = args_json

    if verbose:
        if _USE_RICH:
            _console.print(Panel(
                f"[bold]{name}[/bold]({args_str})",
                title="[cyan]Tool Call[/cyan]",
                border_style="cyan",
                expand=False,
            ))
        else:
            print(f"\n[Tool Call: {name}({args_str})]")
    else:
        brief = args_str if len(args_str) <= 80 else args_str[:77] + "..."
        if _USE_RICH:
            _console.print(f"  [cyan]▶[/cyan] {name}({brief})")
        else:
            print(f"  > {name}({brief})")


def _print_tool_result(name: str, result: str, verbose: bool = False) -> None:
    if verbose:
        preview = result[:300] + "..." if len(result) > 300 else result
        if _USE_RICH:
            _console.print(Panel(
                preview,
                title=f"[green]Result: {name}[/green]",
                border_style="green",
                expand=False,
            ))
        else:
            print(f"[Tool Result: {preview}]\n")
    else:
        if _USE_RICH:
            _console.print(f"  [green]✓[/green] {len(result)} chars")
        else:
            print(f"  ✓ {len(result)} chars")
