import re
from pathlib import Path
from agent.registry import register

MAX_RESULTS = 200


@register({
    "name": "file_search",
    "description": (
        "Search file contents for a pattern (grep-like). "
        "Returns matching lines as 'filepath:line_number: content'. "
        "Supports regular expressions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search string or regular expression",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory)",
            },
            "glob": {
                "type": "string",
                "description": "File filter pattern e.g. '*.md', '*.txt' (default: all files)",
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive search (default: false)",
            },
        },
        "required": ["pattern"],
    },
})
def file_search(
    pattern: str,
    path: str = ".",
    glob: str = "*",
    ignore_case: bool = False,
) -> str:
    try:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid pattern: {e}"

    base = Path(path)
    if not base.exists():
        return f"Error: path not found: {path}"

    files = [base] if base.is_file() else sorted(base.rglob(glob))

    matches = []
    for f in files:
        if not f.is_file():
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{f}:{i}: {line}")
                if len(matches) >= MAX_RESULTS:
                    matches.append(f"[... truncated at {MAX_RESULTS} results ...]")
                    return "\n".join(matches)

    if not matches:
        return "No matches found."
    return "\n".join(matches)
