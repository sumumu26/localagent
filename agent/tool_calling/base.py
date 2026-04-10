from abc import ABC, abstractmethod


class ToolCallingAdapter(ABC):
    """
    モデルごとのツール呼び出しフォーマットを抽象化するインターフェース。
    新しいモデルに対応する場合はこのクラスを継承して実装する。
    """

    @abstractmethod
    def build_system_prompt(self, base_prompt: str, tool_defs: list) -> str:
        """ツール定義をシステムプロンプトに埋め込む。"""
        ...

    @abstractmethod
    def extract_tool_calls(self, text: str) -> list[dict]:
        """モデルの出力テキストからツール呼び出しを抽出する。"""
        ...

    @abstractmethod
    def format_tool_result(self, name: str, result: str) -> str:
        """ツールの実行結果をモデルが読める形式にフォーマットする。"""
        ...

    @abstractmethod
    def strip_tool_calls(self, text: str) -> str:
        """ユーザー向け表示からツール呼び出しブロックを除去する。"""
        ...
