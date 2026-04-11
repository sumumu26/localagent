import sys
import subprocess
import tempfile
from pathlib import Path
from agent.registry import register

DEFAULT_TIMEOUT = 30


@register({
    "name": "python_exec",
    "description": (
        "Execute Python code and return its output (stdout + stderr). "
        "Use for calculations, data processing, or scripting. "
        f"Default timeout is {DEFAULT_TIMEOUT}s."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": f"Execution timeout in seconds (default {DEFAULT_TIMEOUT})",
            },
        },
        "required": ["code"],
    },
})
def python_exec(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name

        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)

    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")

    return "\n".join(parts) if parts else "(no output)"
