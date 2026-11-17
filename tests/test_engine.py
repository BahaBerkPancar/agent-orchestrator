import json

import pytest

from agent_orchestrator.engine import UsageError
from agent_orchestrator.model import (
    Finish,
    MalformedAction,
    ScriptedModel,
    ToolCall,
    parse_action,
)
from agent_orchestrator.policy import Policy
from agent_orchestrator.state import rebuild

from .conftest import event_types, finish, outbox, tc

EMAIL = tc("send_email", to="a@example.com", subject="hi", body="hello")


def test_happy_path_transitions(engine):
    s = engine.start("sum", [tc("calc", expression="1+2"), finish("3")])
    assert s.status == "done" and s.answer == "3"
    assert event_types(engine, s.run_id) == [
        "started",
        "model_call",
        "tool_requested",
        "tool_result",
        "model_call",
        "finished",
    ]
    assert (s.step, s.tool_calls) == (2, 1)
    assert s.scratchpad["last_result"] == "3"


def test_seq_is_monotonic(engine):
    s = engine.start("x", [tc("calc", expression="1"), finish()])
    seqs = [e["seq"] for e in engine.store.events(s.run_id)]
    assert seqs == sorted(set(seqs))


def test_event_log_is_append_only(engine):
    s = engine.start("x", [finish()])
    import sqlite3

    with pytest.raises(sqlite3.DatabaseError):
        engine.store.db.execute("DELETE FROM events")
    with pytest.raises(sqlite3.DatabaseError):
        engine.store.db.execute("UPDATE events SET type='x'")
    assert engine.store.events(s.run_id)


def test_checkpoint_rebuilds_exactly_from_events(engine):
    s = engine.start(
        "x", [tc("calc", expression="1"), EMAIL, {"action": "bogus"}, finish()]
    )
    assert s.status == "waiting_approval"
    assert engine.store.rebuild(s.run_id) == engine.store.load(s.run_id)
    s = engine.decide(s.run_id, "approve")
    assert s.status == "done"
    assert (
        engine.store.rebuild(s.run_id).to_dict()
        == engine.store.load(s.run_id).to_dict()
    )
    assert rebuild(s.run_id, engine.store.events(s.run_id)).to_dict() == s.to_dict()


def test_approve_flow(engine):
    s = engine.start("mail", [EMAIL, finish()])
    assert s.status == "waiting_approval" and s.pending["name"] == "send_email"
    assert outbox(engine, s.run_id) == []  # nothing happened yet
    s = engine.decide(s.run_id, "approve")
    assert s.status == "done"
    assert len(outbox(engine, s.run_id)) == 1


def test_reject_feeds_observation_and_no_side_effect(engine):
    s = engine.start("mail", [EMAIL, finish("gave up")])
    s = engine.decide(s.run_id, "reject", reason="not today")
    assert s.status == "done"
    assert outbox(engine, s.run_id) == []
    assert any(
        "REJECTED" in m["content"] and "not today" in m["content"] for m in s.messages
    )
    assert "tool_result" not in event_types(engine, s.run_id)


def test_edit_flow_and_revalidation(engine):
    s = engine.start("mail", [EMAIL, finish()])
    with pytest.raises(UsageError):
        engine.decide(s.run_id, "edit", {"to": "x"})  # missing fields
    with pytest.raises(UsageError):
        engine.decide(s.run_id, "edit", None)
    assert engine.store.load(s.run_id).status == "waiting_approval"
    edited = {"to": "safe@example.com", "subject": "edited", "body": "b"}
    s = engine.decide(s.run_id, "edit", edited)
    assert s.status == "done"
    assert outbox(engine, s.run_id)[0]["to"] == "safe@example.com"


def test_decide_requires_waiting_run(engine):
    s = engine.start("x", [finish()])
    with pytest.raises(UsageError):
        engine.decide(s.run_id, "approve")


def test_policy_auto_approve_and_auto_reject(make_engine):
    pol = Policy(
        [
            {
                "tool": "send_email",
                "match": {"to": "*@example.com"},
                "decision": "approve",
            },
            {
                "tool": "send_email",
                "match": {"to": "*@evil.test"},
                "decision": "reject",
            },
        ]
    )
    e = make_engine(policy=pol)
    s = e.start("x", [EMAIL, finish()])
    assert s.status == "done" and len(outbox(e, s.run_id)) == 1
    assert "approval_requested" not in event_types(e, s.run_id)
    assert any(
        ev["data"].get("by") == "policy"
        for ev in e.store.events(s.run_id)
        if ev["type"] == "approval_decided"
    )
    bad = tc("send_email", to="x@evil.test", subject="s", body="b")
    s2 = e.start("y", [bad, finish()])
    assert s2.status == "done" and outbox(e, s2.run_id) == []
    # no matching rule -> human
    other = tc("send_email", to="x@other.org", subject="s", body="b")
    assert e.start("z", [other, finish()]).status == "waiting_approval"


def test_policy_toml_loading(tmp_path):
    f = tmp_path / "policy.toml"
    f.write_text(
        '[[rule]]\ntool = "send_email"\nmatch = { to = "*@example.com" }\ndecision = "approve"\n'
    )
    p = Policy.load(f)
    assert p.decide("send_email", {"to": "a@example.com"}) == "approve"
    assert p.decide("send_email", {"to": "a@other.com"}) is None
    assert p.decide("fs_write", {"to": "a@example.com"}) is None
    assert Policy.load(tmp_path / "missing.toml").rules == []
    f.write_text('[[rule]]\ntool = "x"\ndecision = "maybe"\n')
    with pytest.raises(ValueError):
        Policy.load(f)


def test_fs_write_outside_drafts_needs_approval(engine):
    s = engine.start("w", [tc("fs_write", path="notes.txt", content="x"), finish()])
    assert s.status == "waiting_approval"
    s = engine.decide(s.run_id, "approve")
    assert (
        engine.home / "runs" / s.run_id / "workspace" / "notes.txt"
    ).read_text() == "x"


def test_tool_errors_become_observations(engine):
    s = engine.start(
        "x",
        [
            tc("fs_read", path="../../etc/passwd"),
            tc("calc", expression="__import__('os')"),
            finish(),
        ],
    )
    assert s.status == "done"
    results = [
        e["data"] for e in engine.store.events(s.run_id) if e["type"] == "tool_result"
    ]
    assert [r["ok"] for r in results] == [False, False]


# ---- malformed model actions ----


def test_malformed_actions_are_recovered(engine):
    script = [
        "not a dict",
        {"action": "dance"},
        tc("calc", expression="2+2"),
        finish("4"),
    ]
    s = engine.start("x", script)
    assert s.status == "done"
    assert (
        event_types(engine, s.run_id).count("error") == 2 and s.consecutive_errors == 0
    )


def test_three_invalid_actions_in_a_row_fail(engine):
    s = engine.start(
        "x", [{"a": 1}, {"action": "nope"}, tc("nonexistent_tool"), finish()]
    )
    assert s.status == "failed" and "invalid" in s.reason
    assert event_types(engine, s.run_id).count("error") == 3


def test_invalid_args_unknown_tool_and_delegate(engine):
    s = engine.start(
        "x",
        [
            tc("calc"),
            tc("calc", expression=5),
            {"action": "delegate", "task": "t"},
            finish(),
        ],
    )
    assert s.status == "failed"  # three errors in a row
    msgs = [m["content"] for m in s.messages if m["role"] == "system"]
    assert (
        "missing required" in msgs[0]
        and "expected string" in msgs[1]
        and "M2" in msgs[2]
    )


def test_error_streak_resets_after_valid_call(engine):
    s = engine.start(
        "x",
        [{"x": 1}, {"x": 2}, tc("calc", expression="1"), {"x": 3}, {"x": 4}, finish()],
    )
    assert s.status == "done"


def test_script_exhausted_fails(engine):
    s = engine.start("x", [tc("calc", expression="1")])
    assert s.status == "failed" and "exhausted" in s.reason


def test_fail_action(engine):
    s = engine.start("x", [{"action": "fail", "reason": "cannot"}])
    assert s.status == "failed" and s.reason == "cannot"


def test_parse_action_and_model():
    assert parse_action(
        {"action": "tool_call", "name": "calc", "args": {}}
    ) == ToolCall("calc", {})
    assert parse_action({"action": "finish", "answer": 3}) == Finish("3")
    with pytest.raises(MalformedAction):
        ScriptedModel([{"bad": 1}]).next(type("S", (), {"step": 0})())
    assert ScriptedModel([]).count_tokens({"a": "one two three"}) > 2


# ---- budgets ----


def test_budget_max_steps(engine):
    script = [tc("calc", expression="1")] * 10 + [finish()]
    s = engine.start("x", script, {"max_steps": 3})
    assert s.status == "budget_exceeded" and "max_steps" in s.reason
    assert s.step == 3 and s.tool_calls == 3
    assert event_types(engine, s.run_id)[-1] == "budget_hit"


def test_budget_max_tool_calls_blocks_further_tools(engine):
    script = [tc("calc", expression="1")] * 5 + [finish()]
    s = engine.start("x", script, {"max_tool_calls": 2})
    assert s.status == "budget_exceeded" and "max_tool_calls" in s.reason
    assert s.tool_calls == 2
    results = event_types(engine, s.run_id).count("tool_result")
    assert results == 2


def test_budget_max_tool_calls_blocks_side_effect(engine):
    s = engine.start(
        "x", [tc("calc", expression="1"), EMAIL, finish()], {"max_tool_calls": 1}
    )
    assert s.status == "budget_exceeded"
    assert outbox(engine, s.run_id) == []


def test_budget_max_tokens(engine):
    script = [tc("calc", expression="1+1+1+1")] * 10
    s = engine.start("x", script, {"max_tokens": 20})
    assert s.status == "budget_exceeded" and "max_tokens" in s.reason
    assert s.tokens > 0


def test_budget_wall_time(make_engine):
    t = iter(range(0, 1000, 10))
    e = make_engine(clock=lambda: next(t))
    s = e.start("x", [tc("calc", expression="1")] * 5, {"max_wall_seconds": 25})
    assert s.status == "budget_exceeded" and "max_wall" in s.reason


def test_cancel(engine):
    s = engine.start("x", [EMAIL, finish()])
    s = engine.cancel(s.run_id)
    assert s.status == "cancelled"
    assert engine.resume(s.run_id).status == "cancelled"


def test_json_roundtrip_of_state(engine):
    s = engine.start("x", [finish()])
    json.dumps(s.to_dict())


def test_policy_does_not_overmatch_multiple_recipients(engine_with_policy):
    e = engine_with_policy
    evil = {"to": "attacker@evil.com, x@example.com", "subject": "s", "body": "b"}
    s = e.start("x", [tc("send_email", **evil), finish()])
    types = event_types(e, s.run_id)
    assert "error" in types and "approval_decided" not in types
    assert outbox(e, s.run_id) == []


def test_policy_matcher_is_anchored_and_segment_safe():
    p = Policy([{"tool": "t", "match": {"to": "*@example.com"}, "decision": "approve"}])
    for bad in [
        "a@evil.com,b@example.com",
        "a@evil.com\nb@example.com",
        "a@evil.com b@example.com",
        "a@example.com\n",
        "a@example.com.evil.com",
    ]:
        assert p.decide("t", {"to": bad}) is None, bad
    assert p.decide("t", {"to": "a@example.com"}) == "approve"


def test_edit_to_forbidden_is_rejected(make_engine):
    from agent_orchestrator.tools import FORBIDDEN, Tool, default_registry

    reg = default_registry()
    reg.register(
        Tool(
            "danger",
            "x",
            {"type": "object"},
            lambda a: FORBIDDEN if a.get("bad") else "review",
            lambda a, c: "ran",
        )
    )
    e = make_engine(registry=reg)
    s = e.start("x", [tc("danger"), finish()])
    assert s.status == "waiting_approval"
    with pytest.raises(UsageError, match="forbidden"):
        e.decide(s.run_id, "edit", {"bad": True})
    assert e.store.load(s.run_id).status == "waiting_approval"


def test_busy_lock_in_process(engine):
    from agent_orchestrator.engine import RunBusy

    s = engine.start("x", [tc("send_email", to="a@b.c", subject="s", body="b")])
    with engine._locked(s.run_id), pytest.raises(RunBusy):
        engine.resume(s.run_id)


def test_policy_rejects_double_at():
    p = Policy([{"tool": "t", "match": {"to": "*@example.com"}, "decision": "approve"}])
    assert (
        p.decide("t", {"to": "a@evil.com@example.com"}) is None
    )  # glob alone is permissive...
