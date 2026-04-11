import sys
from config import parse_args
from agent.llm import load_model
from agent.loop import run_loop
from agent import registry
from agent.tool_calling import get_adapter
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


def main() -> None:
    cfg = parse_args()

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

            messages.append({"role": "user", "content": user_input})

            response = run_loop(llm, messages, cfg)

            if _USE_RICH:
                _console.print(f"\n[bold green]Assistant:[/bold green] {response}\n")
            else:
                print(f"\nAssistant: {response}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        del llm


if __name__ == "__main__":
    main()
