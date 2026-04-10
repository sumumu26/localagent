"""
Text-based tool calling for ChatML-format models (Qwen3.5).
Instead of relying on llama-cpp-python's native tool_calls JSON parsing,
we format tool definitions in the system prompt and parse <tool_call> tags
from the model's text output.
"""

import json
import re

_TOOL_INSTRUCTIONS = """\

You have access to the following tools:

<tools>
{tools_json}
</tools>

To use a tool, include this in your response:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

You may call multiple tools. After receiving the results, give your final answer."""


def build_system_prompt(base_prompt: str, tool_defs: list) -> str:
    """Append tool definitions to the system prompt."""
    if not tool_defs:
        return base_prompt
    tools_json = json.dumps(
        [d["function"] for d in tool_defs],
        ensure_ascii=False,
        indent=2,
    )
    return base_prompt + _TOOL_INSTRUCTIONS.format(tools_json=tools_json)


def extract_tool_calls(text: str) -> list[dict]:
    """Parse all <tool_call>...</tool_call> blocks from model output."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    calls = []
    for m in re.findall(pattern, text, re.DOTALL):
        try:
            calls.append(json.loads(m))
        except json.JSONDecodeError:
            pass
    return calls


def format_tool_result(name: str, result: str) -> str:
    """Wrap a tool result in <tool_response> tags."""
    payload = json.dumps({"name": name, "result": result}, ensure_ascii=False)
    return f"<tool_response>\n{payload}\n</tool_response>"


def strip_tool_calls(text: str) -> str:
    """Remove <tool_call> blocks from text before displaying to the user."""
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
