from agent.tool_calling.base import ToolCallingAdapter
from agent.tool_calling.qwen import QwenAdapter
from agent.tool_calling.gemma import GemmaAdapter

# chat_format → アダプタのマッピング
_ADAPTER_MAP: dict[str, type[ToolCallingAdapter]] = {
    "gemma": GemmaAdapter,
    "gemma4": GemmaAdapter,
    "gemma-4": GemmaAdapter,
}

_DEFAULT_ADAPTER = QwenAdapter


def get_adapter(chat_format: str) -> ToolCallingAdapter:
    """
    chat_format に応じたツール呼び出しアダプタを返す。

    新しいモデルへの対応手順:
      1. agent/tool_calling/<model>.py を作成し ToolCallingAdapter を継承
      2. _ADAPTER_MAP にエントリを追加
    """
    cls = _ADAPTER_MAP.get(chat_format.lower(), _DEFAULT_ADAPTER)
    return cls()
