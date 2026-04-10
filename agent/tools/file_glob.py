import glob as _glob
from agent.registry import register


@register({
    "name": "file_glob",
    "description": (
        "Find files matching a glob pattern. "
        "Returns a newline-separated list of matching file paths. "
        "Use ** for recursive matching, e.g. 'src/**/*.py'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py' or '/home/user/docs/*.txt'",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether ** matches recursively (default true)",
            },
        },
        "required": ["pattern"],
    },
})
def file_glob(pattern: str, recursive: bool = True) -> str:
    try:
        matches = _glob.glob(pattern, recursive=recursive)
    except Exception as e:
        return f"Error: {e}"

    if not matches:
        return f"No files matched pattern: {pattern}"
    return "\n".join(sorted(matches))
