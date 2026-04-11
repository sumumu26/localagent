from pathlib import Path
from agent.registry import register


@register({
    "name": "file_write",
    "description": (
        "Write text content to a file. "
        "Creates the file (and any missing parent directories) if it does not exist, "
        "or overwrites it if it does. "
        "Use file_read to inspect a file before overwriting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "content": {
                "type": "string",
                "description": "Text content to write",
            },
        },
        "required": ["path", "content"],
    },
})
def file_write(path: str, content: str) -> str:
    p = Path(path)

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"

    return f"Written {len(content)} characters to {path}"
