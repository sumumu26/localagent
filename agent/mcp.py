"""
MCP (Model Context Protocol) クライアント実装。

stdio トランスポート (ローカルサブプロセス) と
Streamable HTTP トランスポート (HTTP/HTTPS リモートサーバ) の両方に対応する。

設定は settings.json の "mcpServers" セクションで行う:

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "env": {}
        },
        "my-remote": {
          "url": "https://example.com/mcp",
          "headers": { "Authorization": "Bearer TOKEN" }
        }
      }
    }

ツール登録名は "{server_name}__{tool_name}" 形式（例: filesystem__read_file）。
"""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path

from agent import registry


# ---------------------------------------------------------------------------
# 基底クラス
# ---------------------------------------------------------------------------

class MCPClientBase(ABC):
    """MCP クライアントの共通インターフェース。"""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def initialize(self) -> None:
        """initialize ハンドシェイクを実行する。"""
        ...

    @abstractmethod
    def list_tools(self) -> list[dict]:
        """サーバが公開するツール一覧を返す。"""
        ...

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """ツールを呼び出し、結果を文字列で返す。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """接続を閉じる / プロセスを終了する。"""
        ...

    def _content_to_text(self, content: list[dict]) -> str:
        """MCP content 配列をテキストに変換する。"""
        parts = []
        for item in content:
            t = item.get("type", "")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                parts.append(f"[image: {item.get('mimeType', 'unknown')}]")
            elif t == "resource":
                res = item.get("resource", {})
                parts.append(res.get("text", f"[resource: {res.get('uri', '')}]"))
        return "\n".join(parts) if parts else "(empty result)"


# ---------------------------------------------------------------------------
# stdio トランスポート
# ---------------------------------------------------------------------------

class StdioMCPClient(MCPClientBase):
    """
    stdio トランスポートで MCP サーバと通信するクライアント。
    サーバはサブプロセスとして起動し、stdin/stdout で JSON-RPC 2.0 メッセージを交換する。
    """

    def __init__(self, name: str, command: str, args: list[str],
                 env: dict | None = None) -> None:
        super().__init__(name)
        merged_env = {**os.environ, **(env or {})}
        self._process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
        )
        self._next_id = 1

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        """JSON-RPC リクエストを送信し、対応するレスポンスを返す。"""
        msg_id = self._next_id
        self._next_id += 1
        msg: dict = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

        # ID が一致するレスポンスが来るまで読み続ける
        while True:
            raw = self._process.stdout.readline()
            if not raw:
                raise RuntimeError(f"MCP server '{self.name}' closed unexpectedly")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue  # ノイズ行は無視
            if data.get("id") == msg_id:
                return data

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        """応答不要の JSON-RPC 通知を送信する。"""
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._process.stdin.write(line.encode())
        self._process.stdin.flush()

    def initialize(self) -> None:
        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hakobune", "version": "0.1.0"},
        })
        self._send_notification("notifications/initialized")

    def list_tools(self) -> list[dict]:
        resp = self._send_request("tools/list")
        if "error" in resp:
            raise RuntimeError(f"tools/list error from '{self.name}': {resp['error']}")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            resp = self._send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
        except RuntimeError as e:
            return f"Error: {e}"
        if "error" in resp:
            err = resp["error"]
            return f"Error: {err.get('message', str(err))}"
        content = resp.get("result", {}).get("content", [])
        return self._content_to_text(content)

    def close(self) -> None:
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()


# ---------------------------------------------------------------------------
# HTTP トランスポート (Streamable HTTP)
# ---------------------------------------------------------------------------

class HttpMCPClient(MCPClientBase):
    """
    Streamable HTTP トランスポートで MCP サーバと通信するクライアント。
    MCP 仕様 2025-03-26 の Streamable HTTP に対応。
    各リクエストは JSON-RPC body を POST し、JSON レスポンスを受け取る。
    initialize 時にサーバが返す Mcp-Session-Id ヘッダを保存して以降のリクエストに付与する。
    """

    def __init__(self, name: str, url: str,
                 headers: dict | None = None) -> None:
        super().__init__(name)
        self._url = url
        self._base_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            self._base_headers.update(headers)
        self._session_id: str | None = None
        self._next_id = 1

    def _post(self, body: dict) -> tuple[dict, dict]:
        """
        JSON-RPC body を POST し、(レスポンス dict, レスポンスヘッダ dict) を返す。
        """
        headers = dict(self._base_headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        payload = json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_headers = dict(resp.headers)
                resp_body = json.loads(resp.read().decode())
                return resp_body, resp_headers
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"HTTP {e.code} from MCP server '{self.name}': {e.reason}"
            )
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to MCP server '{self.name}': {e.reason}"
            )

    def _next_msg_id(self) -> int:
        mid = self._next_id
        self._next_id += 1
        return mid

    def initialize(self) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": self._next_msg_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hakobune", "version": "0.1.0"},
            },
        }
        _, resp_headers = self._post(body)
        # セッション ID をキャッシュ（大文字小文字不定のためケースインセンシティブに取得）
        for key, val in resp_headers.items():
            if key.lower() == "mcp-session-id":
                self._session_id = val
                break

    def list_tools(self) -> list[dict]:
        body = {
            "jsonrpc": "2.0",
            "id": self._next_msg_id(),
            "method": "tools/list",
        }
        resp_body, _ = self._post(body)
        if "error" in resp_body:
            raise RuntimeError(
                f"tools/list error from '{self.name}': {resp_body['error']}"
            )
        return resp_body.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        body = {
            "jsonrpc": "2.0",
            "id": self._next_msg_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        try:
            resp_body, _ = self._post(body)
        except RuntimeError as e:
            return f"Error: {e}"
        if "error" in resp_body:
            err = resp_body["error"]
            return f"Error: {err.get('message', str(err))}"
        content = resp_body.get("result", {}).get("content", [])
        return self._content_to_text(content)

    def close(self) -> None:
        pass  # HTTP は stateless のため特に後処理不要


# ---------------------------------------------------------------------------
# モジュールレベルのクライアント管理
# ---------------------------------------------------------------------------

_clients: list[MCPClientBase] = []


def load_mcp_servers(settings_path: str) -> None:
    """
    settings.json の "mcpServers" セクションを読み込み、
    各サーバを起動してツールをレジストリに登録する。
    サーバの起動に失敗しても警告を出して続行する（エージェント全体は止めない）。
    """
    p = Path(settings_path)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to read {settings_path}: {e}", file=sys.stderr)
        return

    servers: dict = data.get("mcpServers", {})
    if not servers:
        return

    for server_name, server_cfg in servers.items():
        try:
            _start_server(server_name, server_cfg)
        except Exception as e:
            print(
                f"Warning: Failed to start MCP server '{server_name}': {e}",
                file=sys.stderr,
            )


def _start_server(name: str, cfg: dict) -> None:
    """1 台の MCP サーバを起動してツールを登録する。"""
    if "url" in cfg:
        client: MCPClientBase = HttpMCPClient(
            name=name,
            url=cfg["url"],
            headers=cfg.get("headers"),
        )
    elif "command" in cfg:
        client = StdioMCPClient(
            name=name,
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )
    else:
        raise ValueError("mcpServers エントリには 'command' または 'url' が必要です")

    client.initialize()
    tools = client.list_tools()

    for tool in tools:
        _register_mcp_tool(client, tool)

    _clients.append(client)
    print(f"  MCP '{name}': {len(tools)} tool(s) loaded")


def _register_mcp_tool(client: MCPClientBase, tool: dict) -> None:
    """MCP ツールを既存の registry に登録する。"""
    original_name: str = tool["name"]
    registered_name = f"{client.name}__{original_name}"

    schema = {
        "name": registered_name,
        "description": tool.get("description", ""),
        "parameters": tool.get("inputSchema", {
            "type": "object",
            "properties": {},
        }),
    }

    # クロージャでクライアントとツール名をキャプチャ
    _client = client
    _tool_name = original_name

    def fn(**kwargs: object) -> str:
        return _client.call_tool(_tool_name, kwargs)

    registry._TOOLS[registered_name] = {"fn": fn, "schema": schema}


def close_all() -> None:
    """全 MCP サーバプロセス / 接続を終了する。"""
    for client in _clients:
        try:
            client.close()
        except Exception:
            pass
    _clients.clear()
