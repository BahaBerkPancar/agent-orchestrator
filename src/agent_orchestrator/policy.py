"""policy.toml: auto-approve (or auto-reject) review-level tool calls by tool + arg patterns.

[[rule]]
tool = "send_email"
match = { to = "*@example.com" }   # anchored glob on str(args[key])
decision = "approve"               # or "reject"
"""

import re
import tomllib
from pathlib import Path


def _matches(pattern, value):
    """Anchored glob: `*` / `?` never cross whitespace, commas, semicolons or angle brackets,
    so 'a@evil.com, b@example.com' cannot match '*@example.com'."""
    ch = r"[^\s,;<>@]"
    rx = "".join(
        ch + "*" if c == "*" else ch if c == "?" else re.escape(c) for c in pattern
    )
    return re.fullmatch(rx, value) is not None


class Policy:
    def __init__(self, rules=None):
        self.rules = rules or []

    @classmethod
    def load(cls, path):
        if path is None or not Path(path).exists():
            return cls()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        rules = data.get("rule", [])
        for r in rules:
            if r.get("decision") not in ("approve", "reject") or "tool" not in r:
                raise ValueError(f"bad policy rule: {r}")
        return cls(rules)

    def decide(self, tool, args):
        """Return 'approve', 'reject' or None (ask a human). First matching rule wins."""
        for r in self.rules:
            if r["tool"] != tool:
                continue
            if all(
                k in args and _matches(pat, str(args[k]))
                for k, pat in r.get("match", {}).items()
            ):
                return r["decision"]
        return None
