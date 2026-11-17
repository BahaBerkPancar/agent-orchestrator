import json
import os
import signal
import subprocess
import sys
import time

from agent_orchestrator.tools import ToolContext, send_email

from .conftest import SHOP_DB, event_types, outbox, tc

EMAIL_ARGS = {"to": "a@example.com", "subject": "hi", "body": "hello"}


def cli(home, *args, env_extra=None):
    env = {
        **os.environ,
        "AGENT_HOME": str(home),
        "AGENT_SHOP_DB": str(SHOP_DB),
        **(env_extra or {}),
    }
    return subprocess.Popen(
        [sys.executable, "-m", "agent_orchestrator", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_kill_and_resume_without_duplicate_side_effects(tmp_path):
    home = tmp_path / "home"
    policy = tmp_path / "policy.toml"
    policy.write_text('[[rule]]\ntool = "send_email"\ndecision = "approve"\n')
    script = tmp_path / "s.json"
    script.write_text(
        json.dumps(
            [
                tc("calc", expression="1+1"),
                tc("send_email", **EMAIL_ARGS),
                tc("fs_write", path="drafts/after.txt", content="x"),
                tc("calc", expression="2+2"),
                {"action": "finish", "answer": "done"},
            ]
        )
    )
    proc = cli(
        home,
        "run",
        "--goal",
        "g",
        "--script",
        str(script),
        "--policy",
        str(policy),
        env_extra={"AGENT_EVENT_DELAY": "0.4"},
    )
    db = home / "agent.db"
    import sqlite3

    deadline = time.time() + 30
    while (
        time.time() < deadline
    ):  # kill right after the email tool_result is checkpointed
        if db.exists():
            con = sqlite3.connect(db)
            n = con.execute(
                "SELECT count(*) FROM events WHERE type='tool_result' AND data LIKE '%send_email%'"
            ).fetchone()[0]
            con.close()
            if n:
                break
        time.sleep(0.05)
    else:
        proc.kill()
        raise AssertionError("email never executed")
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    assert proc.returncode == -signal.SIGKILL

    from agent_orchestrator.engine import Engine

    e = Engine(home, shop_db=SHOP_DB)
    run_id = e.store.run_ids()[0]
    mid = e.store.load(run_id)
    assert mid.status == "running" and mid.step < 5  # genuinely mid-run
    e.store.close()

    r = cli(home, "resume", run_id)
    out, err = r.communicate(timeout=60)
    assert r.returncode == 0, err
    assert "status=done" in out

    e = Engine(home, shop_db=SHOP_DB)
    ids = [
        ev["data"]["call_id"]
        for ev in e.store.events(run_id)
        if ev["type"] == "tool_result"
    ]
    assert len(ids) == len(set(ids)) == 4  # each tool call executed/recorded once
    assert len(outbox(e, run_id)) == 1
    assert e.store.rebuild(run_id).to_dict() == e.store.load(run_id).to_dict()
    e.store.close()


def test_crash_between_tool_execution_and_result_is_idempotent(engine):
    """Worst window: the side effect happened but its tool_result was never logged."""
    script = [tc("send_email", **EMAIL_ARGS), {"action": "finish", "answer": "ok"}]
    s = engine.start("x", script)
    assert s.status == "waiting_approval"
    ws = engine.home / "runs" / s.run_id / "workspace"
    # simulate: approval logged, tool ran, process died before tool_result
    engine.store.append(
        s.run_id,
        "approval_decided",
        {"call_id": "c1", "decision": "approve", "by": "human", "reason": ""},
    )
    send_email(EMAIL_ARGS, ToolContext(ws, f"{s.run_id}:c1", None))
    s = engine.resume(s.run_id)
    assert s.status == "done"
    assert len(outbox(engine, s.run_id)) == 1
    assert event_types(engine, s.run_id).count("tool_result") == 1


def test_crash_after_model_call_does_not_skip_action(engine):
    """A model_call logged without its follow-up event must be handled, not re-asked."""
    s = engine.start(
        "x",
        [tc("calc", expression="1"), {"action": "finish", "answer": "ok"}],
        {"max_steps": 1},
    )
    assert s.status == "budget_exceeded"
    # fresh run: log started + a model_call only, as if killed right after it
    run_id = "manual"
    script = [tc("calc", expression="40+2"), {"action": "finish", "answer": "42"}]
    engine.store.append(
        run_id,
        "started",
        {
            "goal": "g",
            "script": script,
            "budgets": {},
            "workspace": str(engine.home / "w"),
        },
    )
    (engine.home / "w").mkdir()
    engine.store.append(run_id, "model_call", {"action": script[0], "tokens": 3})
    s = engine.resume(run_id)
    assert s.status == "done" and s.step == 2 and s.tool_calls == 1


def test_resume_on_waiting_run_stays_waiting(engine):
    s = engine.start("x", [tc("send_email", **EMAIL_ARGS)])
    assert engine.resume(s.run_id).status == "waiting_approval"
    assert outbox(engine, s.run_id) == []


def test_waiting_run_survives_process_restart(tmp_path, make_engine):
    e = make_engine()
    s = e.start(
        "x", [tc("send_email", **EMAIL_ARGS), {"action": "finish", "answer": "ok"}]
    )
    r = cli(e.home, "approve", s.run_id, "--decision", "approve")
    _, err = r.communicate(timeout=60)
    assert r.returncode == 0, err
    assert len(outbox(e, s.run_id)) == 1


# ---- locking / races (review round 1) ----


def test_second_process_gets_run_busy(tmp_path):
    home = tmp_path / "home"
    script = tmp_path / "s.json"
    script.write_text(
        json.dumps(
            [tc("calc", expression="1")] * 5 + [{"action": "finish", "answer": "x"}]
        )
    )
    p1 = cli(
        home,
        "run",
        "--goal",
        "g",
        "--script",
        str(script),
        env_extra={"AGENT_EVENT_DELAY": "0.3"},
    )
    from agent_orchestrator.engine import Engine

    deadline = time.time() + 30
    run_id = None
    while time.time() < deadline and not run_id:
        if (home / "agent.db").exists():
            e = Engine(home, shop_db=SHOP_DB)
            ids = e.store.run_ids()
            e.store.close()
            run_id = ids[0] if ids else None
        time.sleep(0.05)
    assert run_id
    p2 = cli(home, "resume", run_id)
    _, err = p2.communicate(timeout=60)
    assert p2.returncode == 7 and "busy" in err
    p1.communicate(timeout=60)
    assert p1.returncode == 0


def test_racing_approvals_execute_email_once(tmp_path):
    from agent_orchestrator.engine import Engine

    home = tmp_path / "home"
    e = Engine(home, shop_db=SHOP_DB)
    s = e.start(
        "x", [tc("send_email", **EMAIL_ARGS), {"action": "finish", "answer": "ok"}]
    )
    e.store.close()
    procs = [
        cli(
            home,
            "approve",
            s.run_id,
            "--decision",
            "approve",
            env_extra={"AGENT_EVENT_DELAY": "0.2"},
        )
        for _ in range(4)
    ]
    codes = [p.wait(timeout=60) for p in procs]
    assert (
        codes.count(0) == 1
    )  # one winner; the rest are busy (7) or find nothing to approve (2)
    assert all(c in (0, 2, 7) for c in codes)
    e = Engine(home, shop_db=SHOP_DB)
    assert len(outbox(e, s.run_id)) == 1
    types = event_types(e, s.run_id)
    assert types.count("approval_decided") == 1 and types.count("tool_result") == 1
    e.store.close()


def test_outbox_dedup_is_atomic_across_processes(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    code = (
        "import sys;from pathlib import Path;from agent_orchestrator.tools import ToolContext,send_email;"
        "send_email({'to':'a@b.c','subject':'s','body':'b'},ToolContext(Path(sys.argv[1]),'run:c1',None))"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, str(ws)]) for _ in range(8)]
    assert [p.wait(timeout=60) for p in procs] == [0] * 8
    assert len((ws / "outbox.jsonl").read_text().splitlines()) == 1


def test_policy_path_is_remembered_by_run(tmp_path):
    from agent_orchestrator.engine import Engine

    pol = tmp_path / "p.toml"
    pol.write_text(
        '[[rule]]\ntool = "send_email"\nmatch = { to = "*@example.com" }\ndecision = "approve"\n'
    )
    e = Engine(tmp_path / "home", shop_db=SHOP_DB)
    s = e.start(
        "x",
        [tc("calc", expression="1"), {"action": "finish", "answer": "ok"}],
        policy_path=pol,
    )
    assert e.store.load(s.run_id).policy_path == str(pol.resolve())
    e.store.close()
