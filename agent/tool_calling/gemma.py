import json
import re
from agent.tool_calling.base import ToolCallingAdapter

# Gemma 4 はシステムプロンプトにツール定義を JSON で渡す
# 出力は <|tool_call|>...<|/tool_call|> ネイティブトークン形式
_TOOL_INSTRUCTIONS = """\

You have access to these tools:

{tools_json}

IMPORTANT RULES:
- For ANY question about facts, people, games, news, or specific details: ALWAYS call a tool first. NEVER answer from memory.
- To call a tool, output exactly:
<|tool_call|>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
<|/tool_call|>
- Wait for the tool result before answering.
- If a search returns no results, try a different query (e.g. English instead of Japanese)."""


class GemmaAdapter(ToolCallingAdapter):
    """
    Gemma 4 向けアダプタ。
    Gemma 4 のネイティブトークン <|tool_call|>...<|/tool_call|> を使用する。
    フォールバックとして <tool_call> XML タグも解釈する。
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
        # Gemma 4 ネイティブトークン形式
        for m in re.findall(r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>", text, re.DOTALL):
            try:
                calls.append(json.loads(m))
            except json.JSONDecodeError:
                pass
        # XML タグ形式（フォールバック）
        if not calls:
            for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
                try:
                    calls.append(json.loads(m))
                except json.JSONDecodeError:
                    pass
        return calls

    def format_tool_result(self, name: str, result: str) -> str:
        payload = json.dumps({"name": name, "result": result}, ensure_ascii=False)
        return f"<|tool_response|>\n{payload}\n<|/tool_response|>"

    def strip_tool_calls(self, text: str) -> str:
        text = re.sub(r"<\|tool_call\|>.*?<\|/tool_call\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        return text.strip()
