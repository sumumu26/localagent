"""
Permission checker compatible with Claude Code's settings.json format.

Rule format: "tool_name(pattern)"
  - pattern uses fnmatch wildcards (* matches anything including newlines)
  - deny rules take precedence over allow rules
  - empty allow list = allow all (unless denied)

Example settings.json:
{
  "permissions": {
    "allow": [
      "python_exec(*)",
      "file_write(src/*)"
    ],
    "deny": [
      "python_exec(*import subprocess*)",
      "file_write(/etc/*)"
    ]
  }
}
"""

import fnmatch
import json
import re
from pathlib import Path


class PermissionDenied(Exception):
    pass


class PermissionChecker:
    def __init__(self):
        self._allow: list[str] = []
        self._deny: list[str] = []

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return  # no settings file = allow all
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to load settings.json: {e}")
        perms = data.get("permissions", {})
        self._allow = perms.get("allow", [])
        self._deny = perms.get("deny", [])

    def check(self, tool_name: str, primary_arg: str) -> None:
        """
        Raises PermissionDenied if the tool call is not permitted.
        primary_arg is matched against the pattern inside tool_name(...).
        Newlines in primary_arg are normalized to spaces for pattern matching.
        """
        normalized = primary_arg.replace("\n", " ").replace("\r", " ")
        call_repr = f"{tool_name}({normalized})"

        # deny takes precedence
        for rule in self._deny:
            if self._match(call_repr, rule):
                raise PermissionDenied(
                    f"Tool call denied by rule '{rule}': {tool_name}({primary_arg[:80]}{'...' if len(primary_arg) > 80 else ''})"
                )

        # if allow list is non-empty, must match at least one rule
        if self._allow:
            for rule in self._allow:
                if self._match(call_repr, rule):
                    return
            raise PermissionDenied(
                f"Tool call not in allow list: {tool_name}({primary_arg[:80]}{'...' if len(primary_arg) > 80 else ''})"
            )

    def _match(self, call_repr: str, rule: str) -> bool:
        """
        Match call_repr against a rule like 'tool_name(pattern)'.
        Falls back to plain fnmatch if rule has no parentheses.
        """
        m = re.fullmatch(r"(\w+)\((.*)\)", rule, re.DOTALL)
        if m:
            rule_tool, rule_pattern = m.group(1), m.group(2)
            tool_match = fnmatch.fnmatch(call_repr.split("(")[0], rule_tool)
            arg_part = call_repr[len(call_repr.split("(")[0]) + 1 : -1]
            return tool_match and fnmatch.fnmatch(arg_part, rule_pattern)
        return fnmatch.fnmatch(call_repr, rule)


# Module-level singleton
_checker = PermissionChecker()


def load(path: str) -> None:
    """Load settings.json from the given path."""
    _checker.load(path)


def check(tool_name: str, primary_arg: str) -> None:
    """Check permission. Raises PermissionDenied if denied."""
    _checker.check(tool_name, primary_arg)
