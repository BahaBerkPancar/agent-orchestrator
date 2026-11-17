import json
import os

import pytest

from agent_orchestrator.schema import validate
from agent_orchestrator.tools import (
    ToolContext,
    ToolError,
    calc,
    default_registry,
    fs_list,
    fs_read,
    fs_write,
    send_email,
    sql_query,
)

from .conftest import SHOP_DB


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ToolContext(ws, "run:c1", SHOP_DB)


def test_fs_roundtrip_and_list(ctx):
    fs_write({"path": "drafts/a.txt", "content": "hi"}, ctx)
    assert fs_read({"path": "drafts/a.txt"}, ctx) == "hi"
    assert "drafts/" in fs_list({}, ctx)


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", ""])
def test_fs_rejects_traversal(ctx, bad):
    with pytest.raises(ToolError):
        fs_read({"path": bad}, ctx)
    with pytest.raises(ToolError):
        fs_write({"path": bad, "content": "x"}, ctx)


def test_fs_rejects_symlink_escape(ctx, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    os.symlink(outside, ctx.workspace / "link")
    os.symlink(outside / "secret.txt", ctx.workspace / "file_link")
    with pytest.raises(ToolError):
        fs_read({"path": "link/secret.txt"}, ctx)
    with pytest.raises(ToolError):
        fs_read({"path": "file_link"}, ctx)
    with pytest.raises(ToolError):
        fs_write({"path": "link/new.txt", "content": "x"}, ctx)
    assert not (outside / "new.txt").exists()
    with pytest.raises(ToolError):
        fs_list({"path": "link"}, ctx)


def test_fs_write_risk_levels():
    t = default_registry().get("fs_write")
    assert t.risk_for({"path": "drafts/a.md"}) == "safe"
    assert t.risk_for({"path": "notes.md"}) == "review"
    assert t.risk_for({"path": "drafts/../notes.md"}) == "review"
    assert t.risk_for({"path": "drafts"}) == "review"


@pytest.mark.parametrize(
    "expr,out",
    [
        ("1+2*3", "7"),
        ("(1+2)*3", "9"),
        ("2**10", "1024"),
        ("-4/2", "-2.0"),
        ("7//2", "3"),
    ],
)
def test_calc_ok(ctx, expr, out):
    assert calc({"expression": expr}, ctx) == out


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id')",
        "open('x')",
        "1+a",
        "[1]",
        "2**9999",
        "1/0",
        "(",
        "'a'*3",
        "True+1",
    ],
)
def test_calc_rejects_code(ctx, expr):
    with pytest.raises(ToolError):
        calc({"expression": expr}, ctx)


def test_sql_select_ok(ctx):
    out = sql_query({"query": "SELECT name FROM customers ORDER BY id LIMIT 1"}, ctx)
    assert "Ada Lovelace" in out


@pytest.mark.parametrize(
    "q",
    [
        "DELETE FROM customers",
        "INSERT INTO customers VALUES (9,'x','y','z')",
        "DROP TABLE orders",
        "UPDATE products SET price=0",
        "SELECT 1; DROP TABLE orders",
        "PRAGMA writable_schema=1",
        "ATTACH DATABASE 'x.db' AS x",
        "WITH x AS (SELECT 1) DELETE FROM orders",
    ],
)
def test_sql_write_attempts_rejected(ctx, q):
    with pytest.raises(ToolError):
        sql_query({"query": q}, ctx)
    assert "Ada" in sql_query({"query": "SELECT name FROM customers"}, ctx)


def test_send_email_only_writes_outbox_and_is_idempotent(ctx):
    a = {"to": "a@b.c", "subject": "s", "body": "b"}
    send_email(a, ctx)
    assert "already" in send_email(a, ctx)
    assert len((ctx.workspace / "outbox.jsonl").read_text().splitlines()) == 1
    send_email(a, ToolContext(ctx.workspace, "run:c2", None))
    assert len((ctx.workspace / "outbox.jsonl").read_text().splitlines()) == 2


def test_schema_validation():
    s = default_registry().get("send_email").schema
    assert validate({"to": "x@y.z", "subject": "s", "body": "b"}, s) == []
    assert validate({"to": "x"}, s)
    assert validate({"to": "x", "subject": "s", "body": "b", "cc": "y"}, s)
    assert validate({"to": 5, "subject": "s", "body": "b"}, s)
    assert validate({"expression": True}, default_registry().get("calc").schema)


def test_registry_unknown_tool():
    _, _, errors = default_registry().check("rm_rf", {})
    assert errors


# ---- hardening (review round 1) ----


@pytest.mark.parametrize(
    "expr",
    [
        "((9**99)**99)**99",
        "(((9**99)**99)**99)**99",
        "9**99**9",
        "(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)*(10**99)",
        "(" + "9999999999*" * 17 + "1)**100",
        "1e308*10",
        "1e308*1e308",
        "(2**1000)**2**2",
        "-(9**99)**99",
    ],
)
def test_calc_big_results_clean_error(ctx, expr):
    with pytest.raises(ToolError):
        calc({"expression": expr}, ctx)


def test_calc_chain_of_multiplications_is_capped(ctx):
    with pytest.raises(ToolError, match="too large"):
        calc({"expression": "*".join(["7**99"] * 30)}, ctx)  # ~8300 bits
    assert calc({"expression": "*".join(["7**99"] * 10)}, ctx) == str(7**990)


def test_calc_big_but_ok(ctx):
    assert calc({"expression": "2**100"}, ctx) == str(2**100)


def test_sql_cross_join_times_out_cleanly(ctx, monkeypatch):
    import time

    from agent_orchestrator import tools

    monkeypatch.setattr(tools, "SQL_TIMEOUT", 0.3)
    t = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        sql_query(
            {
                "query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                "SELECT count(*) FROM c, c AS d, c AS e"
            },
            ctx,
        )
    assert time.monotonic() - t < 5
    t = time.monotonic()
    with pytest.raises(ToolError, match="timed out"):
        sql_query(
            {
                "query": "SELECT count(*) FROM orders a, orders b, orders c, orders d, "
                "orders e, orders f, orders g, orders h, orders i, orders j, orders k, orders l"
            },
            ctx,
        )
    assert time.monotonic() - t < 5


def test_sql_blob_and_row_and_size_caps(ctx):
    out = json.loads(
        sql_query({"query": "SELECT randomblob(16) AS b, zeroblob(10) AS z"}, ctx)
    )
    assert out["rows"][0][0].startswith("<blob 16") and out["truncated"] is False
    big = json.loads(
        sql_query(
            {
                "query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c LIMIT 5000) SELECT x FROM c"
            },
            ctx,
        )
    )
    assert big["truncated"] is True and len(json.dumps(big)) <= 4100
    wide = json.loads(sql_query({"query": "SELECT hex(randomblob(2000))"}, ctx))
    assert wide["truncated"] is True and wide["rows"] == []


def test_send_email_to_must_be_single_address():
    t = default_registry().get("send_email")
    ok = {"subject": "s", "body": "b"}
    assert not validate({**ok, "to": "a@example.com"}, t.schema)
    for bad in [
        "a@evil.com, x@example.com",
        "a@evil.com\nx@example.com",
        "a@b.c;d@e.f",
        "a b@c.d",
        "nobody",
        "<a@b.c>",
    ]:
        assert validate({**ok, "to": bad}, t.schema), bad


def test_schema_nits():
    assert validate(True, {"enum": [1]})
    assert not validate(1, {"enum": [1]})
    assert validate(float("nan"), {"type": "number"})
    assert validate(float("inf"), {"type": "number"})
    assert any("unknown type" in e for e in validate(1, {"type": "strng"}))


def test_outbox_reader_skips_torn_line_and_send_repairs(ctx):
    from agent_orchestrator.tools import read_outbox

    send_email({"to": "a@b.c", "subject": "s", "body": "b"}, ctx)
    p = ctx.workspace / "outbox.jsonl"
    with p.open("a") as f:
        f.write('{"idempotency_key": "run:c9", "to": "x')  # torn write
    assert len(read_outbox(p)) == 1
    send_email(
        {"to": "a@b.c", "subject": "s", "body": "b"},
        ToolContext(ctx.workspace, "run:c3", None),
    )
    assert [r["idempotency_key"] for r in read_outbox(p)] == ["run:c1", "run:c3"]


@pytest.mark.parametrize(
    "q",
    [
        "SELECT zeroblob(999999999)",
        "SELECT randomblob(999999999)",
        "SELECT replace(hex(zeroblob(400000)), '0', hex(zeroblob(400000)))",
        "SELECT hex(zeroblob(600000))",
    ],
)
def test_sql_huge_values_fail_fast(ctx, q):
    import time

    t = time.monotonic()
    with pytest.raises(ToolError):
        sql_query({"query": q}, ctx)
    assert time.monotonic() - t < 2


def test_address_local_part_excludes_at():
    s = default_registry().get("send_email").schema
    assert validate({"to": "a@evil.com@example.com", "subject": "s", "body": "b"}, s)


def test_calc_rejects_complex(ctx):
    for e in ["(-8)**0.5", "(-1)**0.5"]:
        with pytest.raises(ToolError):
            calc({"expression": e}, ctx)


def test_sql_printf_huge_returns_null_fast(ctx):
    import time

    t = time.monotonic()
    out = json.loads(sql_query({"query": "SELECT printf('%.*c', 999999999, 'x')"}, ctx))
    assert out["rows"] == [[None]] and time.monotonic() - t < 2
