"""Tool registry, risk levels and the built-in tools. Nothing here touches anything
outside the per-run workspace (and a read-only demo database)."""

import ast
import fcntl
import json
import math
import operator
import os
import posixpath
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .schema import validate

SAFE, REVIEW, FORBIDDEN = "safe", "review", "forbidden"
MAX_OUTPUT = 4000


class ToolError(Exception):
    """Expected failure; reported to the model as an observation."""


@dataclass
class ToolContext:
    workspace: Path
    idempotency_key: str
    shop_db: Path | None = None


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    risk: str | Callable[[dict], str]
    execute: Callable[[dict, ToolContext], str]

    def risk_for(self, args):
        return self.risk(args) if callable(self.risk) else self.risk


class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools.get(name)

    def check(self, name, args):
        """Return (tool, risk, errors)."""
        tool = self.tools.get(name)
        if tool is None:
            return None, None, [f"unknown tool '{name}'"]
        errors = validate(args, tool.schema)
        return tool, (tool.risk_for(args) if not errors else None), errors


# ---- workspace confinement -------------------------------------------------


def safe_path(workspace: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or "\0" in rel:
        raise ToolError("invalid path")
    if os.path.isabs(rel):
        raise ToolError("absolute paths are not allowed")
    root = workspace.resolve()
    target = (root / rel).resolve()  # follows symlinks, collapses '..'
    if target != root and root not in target.parents:
        raise ToolError("path escapes the workspace")
    return target


def fs_read(args, ctx):
    p = safe_path(ctx.workspace, args["path"])
    if not p.is_file():
        raise ToolError("not a file")
    return p.read_text(errors="replace")[:MAX_OUTPUT]


def fs_write(args, ctx):
    p = safe_path(ctx.workspace, args["path"])
    if p.is_dir():
        raise ToolError("path is a directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    # re-check after mkdir in case a symlinked parent was created concurrently
    safe_path(ctx.workspace, args["path"])
    p.write_text(args["content"])
    return f"wrote {len(args['content'])} chars to {args['path']}"


def fs_write_risk(args):
    path = args.get("path")
    if not isinstance(path, str):
        return REVIEW
    norm = posixpath.normpath(path)
    parts = norm.split("/")
    return (
        SAFE
        if len(parts) > 1 and parts[0] == "drafts" and ".." not in parts
        else REVIEW
    )


def fs_list(args, ctx):
    p = safe_path(ctx.workspace, args.get("path", "."))
    if not p.is_dir():
        raise ToolError("not a directory")
    return json.dumps(sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir()))


# ---- calc (no eval) --------------------------------------------------------

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


MAX_BITS = 4096


def _check(v):
    if isinstance(v, complex):
        raise ToolError("result is not a real number")
    if isinstance(v, int) and v.bit_length() > MAX_BITS:
        raise ToolError("result too large")
    if isinstance(v, float) and not math.isfinite(v):
        raise ToolError("result is not a finite number")
    return v


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _check(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ToolError("exponent too large")
            if (
                isinstance(left, int)
                and isinstance(right, int)
                and right > 0
                and left.bit_length() * right > MAX_BITS
            ):
                raise ToolError("result too large")  # checked before computing
        return _check(_BIN[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _check(_UN[type(node.op)](_eval(node.operand)))
    raise ToolError("only numbers and + - * / // % ** ( ) are allowed")


def calc(args, ctx):
    expr = args["expression"]
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        return str(_eval(tree))
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except (SyntaxError, ValueError, OverflowError, RecursionError, MemoryError):
        raise ToolError("invalid expression") from None


# ---- sql_query (read-only) -------------------------------------------------

_SQL_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _SQL_ALLOWED else sqlite3.SQLITE_DENY


SQL_TIMEOUT = 2.0  # seconds
SQL_MAX_ROWS = 1000
SQL_MAX_VALUE_BYTES = 1_000_000


def _cell(v):
    return f"<blob {len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else v


def sql_query(args, ctx):
    sql = args["query"].strip().rstrip(";").strip()
    if not sql.lower().startswith(("select", "with")) or ";" in sql:
        raise ToolError("only a single SELECT statement is allowed")
    if ctx.shop_db is None or not Path(ctx.shop_db).exists():
        raise ToolError("shop database not available")
    con = sqlite3.connect(f"file:{ctx.shop_db}?mode=ro", uri=True)
    # cap any single string/blob (zeroblob, randomblob, printf, replicate) before it is allocated
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, SQL_MAX_VALUE_BYTES)
    deadline = time.monotonic() + SQL_TIMEOUT
    try:
        con.execute("PRAGMA query_only=ON")
        con.set_authorizer(_authorizer)
        # a non-zero return aborts the running statement
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description]
        fetched = cur.fetchmany(SQL_MAX_ROWS + 1)
        truncated = len(fetched) > SQL_MAX_ROWS
        rows = []
        for row in fetched[:SQL_MAX_ROWS]:
            rows.append([_cell(v) for v in row])
            if len(json.dumps({"columns": cols, "rows": rows})) > MAX_OUTPUT:
                rows.pop()
                truncated = True
                break
        return json.dumps({"columns": cols, "rows": rows, "truncated": truncated})
    except sqlite3.Error as e:
        if "interrupt" in str(e).lower():
            raise ToolError(f"query timed out after {SQL_TIMEOUT}s") from None
        raise ToolError(f"sql error: {e}") from None
    finally:
        con.close()


# ---- send_email (outbox file only) -----------------------------------------


def read_outbox(path):
    """Parse outbox lines; a partially written (or corrupt) line is skipped."""
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def send_email(args, ctx):
    outbox = safe_path(ctx.workspace, "outbox.jsonl")
    record = {
        "idempotency_key": ctx.idempotency_key,
        "to": args["to"],
        "subject": args["subject"],
        "body": args["body"],
    }
    line = (json.dumps(record) + "\n").encode()
    # Exclusive file lock makes check-then-append atomic; O_APPEND + one write keeps lines whole.
    fd = os.open(outbox, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        existing = b""
        while chunk := os.read(fd, 65536):
            existing += chunk
        for raw in existing.decode(errors="replace").splitlines():
            try:
                if json.loads(raw).get("idempotency_key") == ctx.idempotency_key:
                    return "email already in outbox (idempotent replay)"
            except (json.JSONDecodeError, AttributeError):
                continue
        if existing and not existing.endswith(b"\n"):
            line = b"\n" + line  # repair a torn last line so ours starts fresh
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)  # also releases the flock
    return f"queued email to {args['to']} in outbox"


def _obj(props, required):
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_S = {"type": "string", "minLength": 1}
_ADDR = {
    "type": "string",
    "maxLength": 254,
    "pattern": r"[^\s,;<>()\"'@]+@[^\s,;<>()\"'@]+",
}


def default_registry() -> Registry:
    r = Registry()
    r.register(
        Tool(
            "fs_read",
            "Read a text file in the workspace.",
            _obj({"path": _S}, ["path"]),
            SAFE,
            fs_read,
        )
    )
    r.register(
        Tool(
            "fs_list",
            "List a workspace directory.",
            _obj({"path": {"type": "string"}}, []),
            SAFE,
            fs_list,
        )
    )
    r.register(
        Tool(
            "fs_write",
            "Write a file in the workspace (outside drafts/ needs approval).",
            _obj(
                {"path": _S, "content": {"type": "string", "maxLength": 100000}},
                ["path", "content"],
            ),
            fs_write_risk,
            fs_write,
        )
    )
    r.register(
        Tool(
            "calc",
            "Evaluate an arithmetic expression.",
            _obj({"expression": _S}, ["expression"]),
            SAFE,
            calc,
        )
    )
    r.register(
        Tool(
            "sql_query",
            "Run one read-only SELECT on the shop demo database.",
            _obj({"query": _S}, ["query"]),
            SAFE,
            sql_query,
        )
    )
    r.register(
        Tool(
            "send_email",
            "Queue an email (writes workspace/outbox.jsonl; never sends).",
            _obj(
                {"to": _ADDR, "subject": _S, "body": {"type": "string"}},
                ["to", "subject", "body"],
            ),
            REVIEW,
            send_email,
        )
    )
    return r
