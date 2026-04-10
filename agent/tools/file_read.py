from pathlib import Path
from agent.registry import register

MAX_READ_BYTES = 32_768  # 32 KB hard cap to protect context window


@register({
    "name": "file_read",
    "description": (
        "Read the contents of a file. "
        "Large files are truncated at 32 KB unless you specify a line range. "
        "Use start_line and end_line to read a specific section."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based line number to start reading from (default: 1)",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based line number to stop reading at, inclusive (optional)",
            },
        },
        "required": ["path"],
    },
})
def file_read(path: str, start_line: int = 1, end_line: int = None) -> str:
    p = Path(path)

    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a file: {path}"

    try:
        raw = p.read_bytes()
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

    # If no line range specified and file is large, truncate by bytes
    if start_line == 1 and end_line is None and len(raw) > MAX_READ_BYTES:
        text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        return text + f"\n\n[... truncated at {MAX_READ_BYTES} bytes — use start_line/end_line to read more ...]"

    try:
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Error decoding file: {e}"

    # Apply line range (1-based, inclusive)
    selected = lines[start_line - 1 : end_line]
    return "\n".join(selected)
