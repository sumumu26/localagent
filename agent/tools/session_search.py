from agent.registry import register
from agent.session import get_current_session, load_session

_MAX_RESULTS = 5
_CONTEXT_WINDOW = 1  # マッチしたメッセージの前後何件を含めるか


@register({
    "name": "session_search",
    "description": (
        "現在のセッションの全会話履歴をキーワード検索し、関連するやり取りを返す。"
        "コンテキスト圧縮で要約された過去の詳細情報を調べるときに使う。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索キーワードまたはフレーズ",
            },
        },
        "required": ["query"],
    },
})
def session_search(query: str) -> str:
    path = get_current_session()
    if not path:
        return "セッションファイルが設定されていません。"

    messages = load_session(path)
    if not messages:
        return "セッション履歴が空です。"

    query_lower = query.lower()
    matched_indices = set()

    for i, msg in enumerate(messages):
        content = msg.get("content", "") or ""
        if query_lower in content.lower():
            for j in range(
                max(0, i - _CONTEXT_WINDOW),
                min(len(messages), i + _CONTEXT_WINDOW + 1),
            ):
                matched_indices.add(j)

    if not matched_indices:
        return f"'{query}' に関連する会話が見つかりませんでした。"

    blocks = []
    sorted_indices = sorted(matched_indices)
    # 連続するインデックスをグループ化して区切り線を挿入
    prev = None
    current_block = []
    for idx in sorted_indices:
        if prev is not None and idx > prev + 1:
            blocks.append(current_block)
            current_block = []
        current_block.append(idx)
        prev = idx
    if current_block:
        blocks.append(current_block)

    results = []
    for block in blocks[:_MAX_RESULTS]:
        for idx in block:
            msg = messages[idx]
            role = msg["role"]
            content = (msg.get("content", "") or "").strip()
            results.append(f"[{role}]\n{content}")
        results.append("---")

    return "\n\n".join(results)
