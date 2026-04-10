import sys
from config import parse_args
from agent.llm import load_model
from agent.loop import run_loop
from agent import registry, tool_calling
import agent.tools  # noqa: F401 — triggers all @register decorators

try:
    from rich.console import Console
    from rich.prompt import Prompt
    _console = Console()
    _USE_RICH = True
except ImportError:
    _USE_RICH = False


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

    # Embed tool definitions into the system prompt
    system_content = tool_calling.build_system_prompt(
        cfg.system_prompt, registry.get_tool_definitions()
    )
    messages = [{"role": "system", "content": system_content}]

    try:
        while True:
            try:
                if _USE_RICH:
                    user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
                else:
                    user_input = input("You: ").strip()
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
