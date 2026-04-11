import json
from typing import Callable
from agent.permissions import check as permission_check, PermissionDenied

_TOOLS: dict[str, dict] = {}


def register(schema: dict):
    """
    Decorator that registers a tool function with its JSON Schema definition.

    Usage:
        @register({
            "name": "my_tool",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {"param": {"type": "string", "description": "..."}},
                "required": ["param"]
            }
        })
        def my_tool(param: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        _TOOLS[schema["name"]] = {"fn": fn, "schema": schema}
        return fn
    return decorator


def get_tool_definitions() -> list[dict]:
    """Returns tool list in the format llama-cpp-python expects."""
    return [
        {"type": "function", "function": entry["schema"]}
        for entry in _TOOLS.values()
    ]


def dispatch(name: str, arguments_json: str) -> str:
    """
    Execute a tool by name. Always returns a string — never raises.
    Errors are returned as strings so the LLM can self-correct.
    """
    if name not in _TOOLS:
        return f"Error: unknown tool '{name}'. Available tools: {list(_TOOLS.keys())}"
    try:
        args = json.loads(arguments_json) if arguments_json else {}

        # Permission check — use first argument value as primary_arg
        primary_arg = str(next(iter(args.values()), ""))
        try:
            permission_check(name, primary_arg)
        except PermissionDenied as e:
            return f"Error: {e}"

        result = _TOOLS[name]["fn"](**args)
        return str(result)
    except json.JSONDecodeError as e:
        return f"Error: could not parse tool arguments as JSON: {e}"
    except TypeError as e:
        return f"Error: wrong arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error: tool '{name}' raised {type(e).__name__}: {e}"
