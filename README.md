# local_agent

GGUFモデルをローカルで直接実行し、ツールを使いながら回答するCLIエージェント。  
フレームワーク不使用のシンプルなReActループ実装。

## 特徴

- **完全ローカル** — llama-cpp-pythonでGGUFモデルを直接読み込み、外部サーバー不要
- **ツール呼び出し対応** — Web検索・ファイル検索・ファイル読み込みを標準装備
- **拡張しやすい** — ツールの追加は1ファイル + 1行のimportだけ
- **依存最小** — LangChainなし、シンプルなReActループをスクラッチ実装

## 必要環境

- Python 3.10+
- ツール呼び出し対応のGGUFモデル（推奨: Qwen3.5-9B-Instruct Q4_K_M）

## セットアップ

```bash
pip install -r requirements.txt
```

GPU（CUDA）を使う場合は llama-cpp-python を先にビルドしてからインストール：

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
pip install -r requirements.txt
```

## 使い方

```bash
python main.py --model /path/to/model.gguf
```

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--model` | 必須 | GGUFモデルのパス |
| `--chat-format` | `auto` | チャットフォーマット（autoはGGUFメタデータから自動検出） |
| `--n-ctx` | `8192` | コンテキスト長 |
| `--n-gpu-layers` | `-1` | GPUオフロード数（-1=全レイヤー） |
| `--temperature` | `0.0` | 生成温度 |
| `--max-tokens` | `1024` | 1回の応答の最大トークン数 |
| `--max-iterations` | `10` | ReActループの最大反復数 |
| `--system-prompt` | — | システムプロンプトの上書き |
| `--verbose` | off | llama-cpp-python の詳細ログを表示 |

### 実行例

```bash
# Qwen3.5-9B Q4_K_M（推奨）
python main.py \
  --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  --n-ctx 8192 \
  --n-gpu-layers -1

# CPUのみ
python main.py \
  --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  --n-gpu-layers 0
```

> **Qwen3.5 の注意点**: thinking mode を持つため、デフォルトのシステムプロンプトに `/no_think` を付与しています。tool calling の安定性のためそのままの使用を推奨します。他のモデルを使う場合は `--system-prompt` で上書きしてください。

起動後はターミナルで対話できます。`exit` または Ctrl-C で終了。

## 標準ツール

| ツール | 説明 |
|---|---|
| `web_search` | DuckDuckGoでWeb検索。タイトル・URL・スニペットを返す |
| `file_glob` | globパターンでファイルを検索（`**/*.py` など再帰対応） |
| `file_read` | ファイルを読み込む。32KBでキャップ、行範囲指定（`start_line`/`end_line`）対応 |

## ツールの追加方法

**1. `agent/tools/my_tool.py` を作成**

```python
from agent.registry import register

@register({
    "name": "my_tool",
    "description": "このツールが何をするかLLMが読む説明文",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "パラメータの説明"},
        },
        "required": ["param1"]
    }
})
def my_tool(param1: str) -> str:
    # 実装
    return result_as_string
```

**2. `agent/tools/__init__.py` に1行追加**

```python
from agent.tools import my_tool
```

以上で完了。レジストリが自動的にLLMへのスキーマ提供と実行ディスパッチを担います。

## プロジェクト構成

```
local_agent/
├── main.py              # CLIエントリポイント・REPLループ
├── config.py            # 設定dataclass + argparse
├── requirements.txt
└── agent/
    ├── llm.py           # llama-cpp-pythonラッパー
    ├── registry.py      # ツールレジストリ（register/dispatch）
    ├── loop.py          # ReActループ本体
    └── tools/
        ├── __init__.py  # ツール登録トリガー
        ├── web_search.py
        ├── file_glob.py
        └── file_read.py
```

## 動作確認済みモデル

tool callingに対応したモデルであれば動作します。

| モデル | GGUF配布元 | 備考 |
|---|---|---|
| **Qwen3.5-9B-Instruct Q4_K_M**（推奨） | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | VRAM 約8GB |
| Qwen3.5-4B-Instruct Q4_K_M | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | VRAM 約4GB、軽量優先の場合 |

## ライセンス

MIT
