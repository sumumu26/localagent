import json
from agent import registry
from agent.llm import chat_completion
from config import Config

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False


def run_loop(llm, messages: list, cfg: Config) -> str:
    """
    Runs the ReAct loop. Mutates `messages` in place.
    Returns the final text response.
    """
    tools = registry.get_tool_definitions()

    for iteration in range(cfg.max_iterations):
        response = chat_completion(llm, messages, tools, cfg)

        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")

        # Assistant message MUST be appended before tool result messages
        messages.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": tool_calls,
        })

        if not tool_calls:
            return msg.get("content") or ""

        # Execute all tool calls (models like Llama 3.1 can request multiple at once)
        for tc in tool_calls:
            call_id = tc["id"]
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", "{}")

            _print_tool_call(fn_name, fn_args)
            result = registry.dispatch(fn_name, fn_args)
            _print_tool_result(fn_name, result)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": fn_name,
                "content": result,
            })

    return (
        f"[Agent stopped after {cfg.max_iterations} iterations. "
        "The last tool results are in the conversation history.]"
    )


def _print_tool_call(name: str, args_json: str) -> None:
    try:
        args = json.loads(args_json)
        args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = args_json

    if _USE_RICH:
        _console.print(Panel(
            f"[bold]{name}[/bold]({args_str})",
            title="[cyan]Tool Call[/cyan]",
            border_style="cyan",
            expand=False,
        ))
    else:
        print(f"\n[Tool Call: {name}({args_str})]")


def _print_tool_result(name: str, result: str) -> None:
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
