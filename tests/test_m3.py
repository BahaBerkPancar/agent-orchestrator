import json
import subprocess
import sys
from urllib.parse import urlencode

import pytest

from agent_orchestrator.cli import main
from agent_orchestrator.engine import UsageError
from agent_orchestrator.model import parse_action
from agent_orchestrator.policy import Policy
from agent_orchestrator.replay import (
    assert_recorded_fixture,
    compare_events,
    recording_from_store,
    replay_recording,
    replay_run,
    write_regression_test,
)
from agent_orchestrator.tools import (
    SAFE,
    Tool,
    ToolContext,
    default_registry,
    send_email,
)
from agent_orchestrator.web import MAX_SESSIONS, WebApplication, parse_content_length

from .conftest import SHOP_DB, finish, outbox, tc

EMAIL = tc("send_email", to="a@example.com", subject="hello", body="body")
GOOD_HOST = {"Host": "127.0.0.1:8070"}


@pytest.mark.parametrize(
    ("decision", "args"),
    [
        ("approve", None),
        ("reject", None),
        (
            "edit",
            {"to": "edited@example.com", "subject": "changed", "body": "body"},
        ),
    ],
)
def test_replay_is_identical_for_each_approval_decision(make_engine, decision, args):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish("finished")])
    state = engine.decide(state.run_id, decision, args, "recorded reason")
    result = replay_run(engine.store, state.run_id, shop_db=SHOP_DB)
    assert result.identical, json.dumps(
        {"original": result.original, "replayed": result.replayed}, sort_keys=True
    )
    assert result.state.status == "done" and result.state.answer == "finished"


def test_replay_preserves_policy_approval_without_human_pause(make_engine):
    engine = make_engine(
        policy=Policy(
            [{"tool": "send_email", "match": {}, "decision": "approve"}]
        )
    )
    state = engine.start("mail", [EMAIL, finish("finished")])
    result = replay_run(engine.store, state.run_id, shop_db=SHOP_DB)
    assert result.identical, (result.original, result.replayed)


class RoutingModel:
    def __init__(self, scripts):
        self.scripts = scripts
        self.last_usage = {}

    def next(self, state):
        return parse_action(self.scripts[state.goal][state.step])

    def count_tokens(self, raw):
        return len(json.dumps(raw).split())


def test_replay_includes_delegated_child(make_engine):
    scripts = {
        "lead": [
            {
                "action": "delegate",
                "task": "worker",
                "tools": ["calc"],
                "budgets": {"max_steps": 3},
            },
            finish("lead done"),
        ],
        "worker": [tc("calc", expression="40+2"), finish("42")],
    }
    engine = make_engine(model=RoutingModel(scripts))
    state = engine.start("lead", allowed_tools=["calc"])
    recording = recording_from_store(engine.store, state.run_id)
    assert len({event["run_id"] for event in recording["events"]}) == 2
    result = replay_run(engine.store, state.run_id, shop_db=SHOP_DB)
    assert result.identical, json.dumps(
        {"original": result.original, "replayed": result.replayed}, sort_keys=True
    )
    assert result.state.answer == "lead done"


def test_replay_of_crash_resumed_run_is_identical(make_engine):
    engine = make_engine()
    run_id = "resumed"
    workspace = engine.home / "runs" / run_id / "workspace"
    workspace.mkdir(parents=True)
    script = [tc("calc", expression="20+22"), finish("42")]
    engine.store.append(
        run_id,
        "started",
        {
            "goal": "resume me",
            "script": script,
            "budgets": {},
            "workspace": str(workspace),
            "workspace_seed": {},
            "allowed_tools": list(engine.registry.tools),
            "model_config": {"type": "scripted"},
        },
    )
    engine.store.append(run_id, "model_call", {"action": script[0], "tokens": 5})
    assert engine.resume(run_id).status == "done"
    result = replay_run(engine.store, run_id, shop_db=SHOP_DB)
    assert result.identical


def test_replay_normalizes_idempotent_outbox_crash_window(make_engine):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish("sent")])
    engine.store.append(
        state.run_id,
        "approval_decided",
        {
            "call_id": "c1",
            "decision": "approve",
            "by": "human",
            "reason": "before crash",
        },
    )
    workspace = engine.home / "runs" / state.run_id / "workspace"
    send_email(
        EMAIL["args"],
        ToolContext(workspace, f"{state.run_id}:c1", SHOP_DB),
    )
    assert engine.resume(state.run_id).status == "done"
    original_result = next(
        event["data"]["output"]
        for event in engine.store.events(state.run_id)
        if event["type"] == "tool_result"
    )
    assert original_result == "email already in outbox (idempotent replay)"
    result = replay_run(engine.store, state.run_id, shop_db=SHOP_DB)
    assert result.identical, (result.original, result.replayed)


def test_delegated_child_approval_and_parent_resume_replay_identically(make_engine):
    scripts = {
        "lead": [
            {
                "action": "delegate",
                "task": "mail worker",
                "tools": ["send_email"],
            },
            finish("lead done"),
        ],
        "mail worker": [EMAIL, finish("worker done")],
    }
    engine = make_engine(model=RoutingModel(scripts))
    parent = engine.start("lead", allowed_tools=["send_email"])
    child_id = parent.active_child
    assert engine.decide(child_id, "approve", reason="child approval").status == "done"
    parent = engine.resume(parent.run_id)
    assert parent.status == "done"
    assert any(
        event["type"] == "delegation_resumed"
        for event in engine.store.events(parent.run_id)
    )
    result = replay_run(engine.store, parent.run_id, shop_db=SHOP_DB)
    assert result.identical, (result.original, result.replayed)


def test_delegation_resumed_is_crash_boundary_metadata():
    event = {
        "seq": 1,
        "run_id": "root",
        "ts": 0,
        "type": "delegation_resumed",
        "data": {"child_id": "child"},
    }
    assert compare_events([event], [], "root") == (True, "identical", None, None)


def test_replay_reports_changed_tool_result(make_engine, tmp_path):
    engine = make_engine()
    state = engine.start("sum", [tc("calc", expression="1+2"), finish("3")])
    recording = recording_from_store(engine.store, state.run_id)
    registry = default_registry()
    original = registry.get("calc")
    registry.register(
        Tool(
            "calc",
            original.description,
            original.schema,
            SAFE,
            lambda args, context: "changed",
        )
    )
    result = replay_recording(
        recording, tmp_path / "replay", shop_db=SHOP_DB, registry=registry
    )
    assert not result.identical and result.message.startswith("diverged at event")
    assert result.original["result_hash"] != result.replayed["result_hash"]


def test_generated_regression_has_fixture_and_passes(make_engine, tmp_path):
    engine = make_engine()
    state = engine.start("sum", [tc("calc", expression="6*7"), finish("42")])
    path = tmp_path / "tests" / "test_recorded_sum.py"
    fixture = write_regression_test(path, recording_from_store(engine.store, state.run_id))
    assert fixture == path.with_suffix(".json")
    assert "assert_recorded_fixture" in path.read_text()
    assert len(path.read_text().splitlines()) < 10
    assert_recorded_fixture(fixture, tmp_path / "direct", SHOP_DB)
    changed = default_registry()
    calc = changed.get("calc")
    changed.register(
        Tool(
            "calc",
            calc.description,
            calc.schema,
            SAFE,
            lambda args, context: "changed",
        )
    )
    with pytest.raises(AssertionError, match="diverged at event"):
        assert_recorded_fixture(
            fixture, tmp_path / "changed", SHOP_DB, registry=changed
        )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_replay_and_to_test_support_command_home(make_engine, tmp_path, capsys):
    engine = make_engine()
    state = engine.start("sum", [tc("calc", expression="2+3"), finish("5")])
    generated = tmp_path / "regressions" / "test_five.py"
    code = main(
        [
            "--shop-db",
            str(SHOP_DB),
            "replay",
            state.run_id,
            "--home",
            str(engine.home),
            "--to-test",
            str(generated),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0 and "identical" in output and "wrote" in output
    assert generated.is_file() and generated.with_suffix(".json").is_file()


def _session(app):
    response = app.handle("GET", "/", GOOD_HOST)
    cookie = response.headers["Set-Cookie"].split(";", 1)[0]
    return cookie, cookie.split("=", 1)[1]


def _post(app, run_id, cookie, csrf, decision, args=None):
    fields = {"csrf": csrf, "decision": decision}
    if args is not None:
        fields["args"] = json.dumps(args)
    return app.handle(
        "POST",
        f"/runs/{run_id}/decision",
        {**GOOD_HOST, "Cookie": cookie},
        urlencode(fields).encode(),
    )


def test_web_lists_details_and_escapes_hostile_args(make_engine):
    engine = make_engine()
    hostile = tc(
        "send_email",
        to="a@example.com",
        subject="<script>alert(1)</script>",
        body="& hostile",
    )
    state = engine.start("hostile <goal>", [hostile, finish()])
    app = WebApplication(engine.home, SHOP_DB)
    cookie, _ = _session(app)
    headers = {**GOOD_HOST, "Cookie": cookie}
    listing = app.handle("GET", "/", headers)
    detail = app.handle("GET", f"/runs/{state.run_id}", headers)
    assert listing.status == detail.status == 200
    assert state.run_id.encode() in listing.body
    assert b"&lt;script&gt;alert(1)&lt;/script&gt;" in detail.body
    assert b"<script>alert(1)</script>" not in detail.body
    assert b"hostile &lt;goal&gt;" in detail.body


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_web_approve_and_reject_use_engine_path(make_engine, decision):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish("done")])
    app = WebApplication(engine.home, SHOP_DB)
    cookie, csrf = _session(app)
    response = _post(app, state.run_id, cookie, csrf, decision)
    assert response.status == 303
    updated = engine.store.load(state.run_id)
    assert updated.status == "done"
    assert bool(outbox(engine, state.run_id)) is (decision == "approve")
    event = next(
        event
        for event in engine.store.events(state.run_id)
        if event["type"] == "approval_decided"
    )
    expected_reason = "web rejection" if decision == "reject" else "web approval"
    assert event["data"]["reason"] == expected_reason


def test_web_edit_validates_schema_and_csrf(make_engine):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish("done")])
    app = WebApplication(engine.home, SHOP_DB)
    cookie, csrf = _session(app)
    rejected = _post(app, state.run_id, cookie, "wrong", "approve")
    assert rejected.status == 403
    invalid = _post(app, state.run_id, cookie, csrf, "edit", {"to": "x"})
    assert invalid.status == 400
    assert engine.store.load(state.run_id).status == "waiting_approval"
    edited = {"to": "new@example.com", "subject": "new", "body": "body"}
    accepted = _post(app, state.run_id, cookie, csrf, "edit", edited)
    assert accepted.status == 303
    assert outbox(engine, state.run_id)[0]["to"] == "new@example.com"


def test_web_reports_run_lock_conflict(make_engine):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish()])
    app = WebApplication(engine.home, SHOP_DB)
    cookie, csrf = _session(app)
    with engine._locked(state.run_id):
        response = _post(app, state.run_id, cookie, csrf, "approve")
    assert response.status == 409
    assert engine.store.load(state.run_id).status == "waiting_approval"


def test_web_rejects_bad_host_before_get_or_post(make_engine):
    engine = make_engine()
    state = engine.start("mail", [EMAIL, finish()])
    app = WebApplication(engine.home, SHOP_DB)
    bad = {"Host": "attacker.example:8070"}
    get_response = app.handle("GET", "/", bad)
    post_response = app.handle(
        "POST",
        f"/runs/{state.run_id}/decision",
        bad,
        urlencode({"csrf": "x", "decision": "approve"}).encode(),
    )
    assert get_response.status == post_response.status == 421
    assert "Set-Cookie" not in get_response.headers
    assert not app.sessions
    assert engine.store.load(state.run_id).status == "waiting_approval"


def test_web_accepts_both_exact_loopback_hosts(make_engine):
    engine = make_engine()
    app = WebApplication(engine.home, SHOP_DB)
    assert app.handle("GET", "/", GOOD_HOST).status == 200
    assert app.handle("GET", "/", {"Host": "localhost:8070"}).status == 200


def test_web_caps_sessions_and_validates_content_length(make_engine):
    engine = make_engine()
    app = WebApplication(engine.home, SHOP_DB)
    for _ in range(MAX_SESSIONS + 5):
        app._session(GOOD_HOST)
    assert len(app.sessions) == MAX_SESSIONS
    assert parse_content_length(None) == parse_content_length("0") == 0
    with pytest.raises(UsageError, match="non-negative integer"):
        parse_content_length("nope")
    with pytest.raises(UsageError, match="non-negative integer"):
        parse_content_length("-1")
