import json
import re
from agent.tool_calling.base import ToolCallingAdapter

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


class QwenAdapter(ToolCallingAdapter):
    """
    Qwen3.5 (chatml) 向けアダプタ。
    <tool_call>...</tool_call> XML タグを使用する。
    """

    def build_system_prompt(self, base_prompt: str, tool_defs: list) -> str:
        if not tool_defs:
            return base_prompt
        tools_json = json.dumps(
            [d["function"] for d in tool_defs],
            ensure_ascii=False,
            indent=2,
        )
        return base_prompt + _TOOL_INSTRUCTIONS.format(tools_json=tools_json)

    def extract_tool_calls(self, text: str) -> list[dict]:
        calls = []
        for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
            try:
                calls.append(json.loads(m))
            except json.JSONDecodeError:
                pass
        return calls

    def format_tool_result(self, name: str, result: str) -> str:
        payload = json.dumps({"name": name, "result": result}, ensure_ascii=False)
        return f"<tool_response>\n{payload}\n</tool_response>"

    def strip_tool_calls(self, text: str) -> str:
        return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
