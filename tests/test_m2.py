import json

import pytest

from agent_orchestrator.cli import main
from agent_orchestrator.engine import Engine
from agent_orchestrator.eval import SuiteModel, load_suite, run_eval, summarize, wilson
from agent_orchestrator.model import (
    MalformedAction,
    ModelTransportError,
    OllamaModel,
    parse_action,
)
from agent_orchestrator.state import RunState
from agent_orchestrator.tools import default_registry

from .conftest import SHOP_DB, event_types, outbox


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, payload):
        self.requests.append(payload)
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(action, prompt_tokens=10, output_tokens=5):
    return {
        "response": json.dumps(action) if not isinstance(action, str) else action,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


def planner_state(**changes):
    state = RunState(
        run_id="r",
        goal="calculate",
        allowed_tools=["calc"],
        messages=[{"role": "user", "content": "calculate"}],
    )
    for key, value in changes.items():
        setattr(state, key, value)
    return state


def test_finish_normalization_and_missing_action_error_are_specific():
    assert parse_action({"answer": 42}).answer == "42"
    with pytest.raises(
        MalformedAction,
        match='missing "action" field; expected one of tool_call\\|finish\\|fail\\|delegate',
    ):
        parse_action({"reason": "no action"})


def test_ollama_planner_validates_and_repairs_once():
    transport = FakeTransport(
        [
            response({"action": "tool_call", "name": "calc", "args": {}}),
            response(
                {
                    "action": "tool_call",
                    "name": "calc",
                    "args": {"expression": "6*7"},
                },
                12,
                3,
            ),
        ]
    )
    model = OllamaModel(default_registry(), transport=transport)
    action = model.next(planner_state())
    assert action.to_dict()["args"] == {"expression": "6*7"}
    assert model.last_usage == {"tokens": 30, "repairs": 1, "normalized": []}
    assert len(transport.requests) == 2
    assert "missing required" in transport.requests[1]["prompt"]
    assert transport.requests[0]["format"]["required"] == ["action"]
    assert transport.requests[0]["format"]["properties"]["action"]["enum"] == [
        "tool_call",
        "finish",
        "fail",
        "delegate",
    ]
    assert transport.requests[0]["options"] == {"temperature": 0, "seed": 42}


def test_ollama_planner_fails_after_one_bounded_repair():
    transport = FakeTransport([response("not json"), response({"action": "dance"})])
    model = OllamaModel(default_registry(), transport=transport)
    with pytest.raises(MalformedAction, match="invalid after repair"):
        model.next(planner_state())
    assert model.last_usage["repairs"] == 1 and len(transport.requests) == 2


def test_failed_repair_is_one_recoverable_engine_error(tmp_path):
    transport = FakeTransport(
        [
            response("not json"),
            response({"action": "dance"}),
            response({"answer": 42}),
        ]
    )
    registry = default_registry()
    engine = Engine(
        tmp_path / "home",
        model=OllamaModel(registry, transport=transport),
        registry=registry,
        shop_db=SHOP_DB,
    )
    state = engine.start("x", allowed_tools=["calc"])
    assert state.status == "done" and state.answer == "42"
    assert event_types(engine, state.run_id).count("error") == 1
    assert state.consecutive_errors == 1
    engine.store.close()


def test_implicit_finish_and_nested_tool_shapes_are_normalized_and_logged(tmp_path):
    transport = FakeTransport(
        [
            response(
                {
                    "action": "tool_call",
                    "args": {
                        "tool_id": "calc.expression",
                        "args": {"expression": "6*7"},
                    },
                }
            ),
            response({"answer": 42}),
        ]
    )
    registry = default_registry()
    engine = Engine(
        tmp_path / "home",
        model=OllamaModel(registry, transport=transport),
        registry=registry,
        shop_db=SHOP_DB,
    )
    state = engine.start("x", allowed_tools=["calc"])
    calls = [
        event
        for event in engine.store.events(state.run_id)
        if event["type"] == "model_call"
    ]
    assert state.status == "done" and state.answer == "42"
    assert calls[0]["data"]["action"]["name"] == "calc"
    assert "normalized nested tool_call arguments" in calls[0]["data"]["normalized"]
    assert (
        'normalized missing "action" field to finish' in calls[1]["data"]["normalized"]
    )
    assert "normalized finish answer to string" in calls[1]["data"]["normalized"]
    engine.store.close()


def test_nested_name_shape_and_prompt_sql_schema():
    transport = FakeTransport(
        [
            response(
                {
                    "action": "tool_call",
                    "args": {
                        "name": "sql_query",
                        "args": {"query": "SELECT COUNT(*) FROM customers"},
                    },
                }
            )
        ]
    )
    model = OllamaModel(default_registry(), transport=transport)
    action = model.next(planner_state(allowed_tools=["sql_query"]))
    assert action.name == "sql_query"
    prompt = transport.requests[0]["prompt"]
    assert "customers(id, name, email, city)" in prompt
    assert prompt.index("tool_call") < prompt.index("finish") < prompt.index("fail")


def test_ollama_retries_infra_errors_and_labels_final_failure():
    transport = FakeTransport(
        [
            ModelTransportError("500 one"),
            ModelTransportError("500 two"),
            response({"answer": "ok"}),
        ]
    )
    model = OllamaModel(
        default_registry(), transport=transport, retry_backoff=0, sleep=lambda _: None
    )
    assert model.next(planner_state()).answer == "ok"
    assert len(transport.requests) == 3

    failing = FakeTransport([ModelTransportError("500")] * 3)
    model = OllamaModel(
        default_registry(), transport=failing, retry_backoff=0, sleep=lambda _: None
    )
    with pytest.raises(ModelTransportError, match="infra error after 3 attempts"):
        model.next(planner_state())


def test_planner_repair_and_tokens_are_recorded(tmp_path):
    transport = FakeTransport(
        [
            response({"action": "tool_call", "name": "calc", "args": {}}),
            response({"action": "finish", "answer": "repaired"}),
        ]
    )
    registry = default_registry()
    engine = Engine(
        tmp_path / "home",
        model=OllamaModel(registry, transport=transport),
        registry=registry,
        shop_db=SHOP_DB,
    )
    state = engine.start("x", allowed_tools=["calc"])
    assert state.status == "done" and state.repairs == 1 and state.tokens == 30
    call = next(
        e for e in engine.store.events(state.run_id) if e["type"] == "model_call"
    )
    assert call["data"]["repairs"] == 1
    engine.store.close()


class RoutingModel:
    def __init__(self, scripts):
        self.scripts = scripts
        self.last_usage = {}

    def next(self, state):
        script = self.scripts[state.goal]
        return parse_action(script[state.step])

    def count_tokens(self, raw):
        return len(json.dumps(raw).split())


def test_delegation_scope_and_budget_carve_out(tmp_path):
    task = "worker calc"
    scripts = {
        "lead": [
            {
                "action": "delegate",
                "task": task,
                "tools": ["calc"],
                "budgets": {
                    "max_steps": 999,
                    "max_tool_calls": 999,
                    "max_tokens": 99999,
                },
            },
            {"action": "finish", "answer": "42"},
        ],
        task: [
            {"action": "tool_call", "name": "calc", "args": {"expression": "40+2"}},
            {"action": "finish", "answer": "42"},
        ],
    }
    engine = Engine(tmp_path / "home", model=RoutingModel(scripts), shop_db=SHOP_DB)
    state = engine.start(
        "lead",
        budgets={"max_steps": 6, "max_tool_calls": 3, "max_tokens": 200},
        allowed_tools=["calc"],
    )
    child = next(
        engine.store.load(run_id)
        for run_id in engine.store.run_ids()
        if run_id != state.run_id
    )
    assert state.status == "done" and child.parent_id == state.run_id
    assert child.role == "worker" and child.allowed_tools == ["calc"]
    assert child.budgets["max_steps"] <= 5
    assert child.budgets["max_tool_calls"] <= 3
    assert child.budgets["max_tokens"] < 200
    engine.store.close()


def test_delegation_wall_budget_uses_parent_remaining_time(tmp_path):
    now = [0.0]
    task = "worker calc"

    class AdvancingModel(RoutingModel):
        def next(self, state):
            if state.goal == "lead" and state.step == 0:
                now[0] = 7.0
            return super().next(state)

    scripts = {
        "lead": [
            {
                "action": "delegate",
                "task": task,
                "tools": ["calc"],
                "budgets": {"max_wall_seconds": 99},
            },
            {"action": "finish", "answer": "done"},
        ],
        task: [{"action": "finish", "answer": "done"}],
    }
    engine = Engine(
        tmp_path / "home",
        model=AdvancingModel(scripts),
        shop_db=SHOP_DB,
        clock=lambda: now[0],
    )
    state = engine.start("lead", budgets={"max_wall_seconds": 10})
    child = next(
        engine.store.load(run_id)
        for run_id in engine.store.run_ids()
        if run_id != state.run_id
    )
    assert child.budgets["max_wall_seconds"] == 3
    engine.store.close()


def test_delegation_cannot_widen_or_include_forbidden(tmp_path):
    scripts = {
        "lead": [
            {"action": "delegate", "task": "bad", "tools": ["fs_write"]},
            {"action": "delegate", "task": "bad", "tools": ["network_request"]},
            {"action": "fail", "reason": "refused"},
        ]
    }
    engine = Engine(tmp_path / "home", model=RoutingModel(scripts), shop_db=SHOP_DB)
    state = engine.start("lead", allowed_tools=["calc", "network_request"])
    errors = [
        event["data"]["message"]
        for event in engine.store.events(state.run_id)
        if event["type"] == "error"
    ]
    assert state.status == "failed"
    assert "subset" in errors[0] and "forbidden" in errors[1]
    assert len(engine.store.run_ids()) == 1
    engine.store.close()


def test_child_approval_pauses_parent_and_parent_decision_resumes_both(tmp_path):
    task = "queue mail"
    scripts = {
        "lead": [
            {"action": "delegate", "task": task, "tools": ["send_email"]},
            {"action": "finish", "answer": "done"},
        ],
        task: [
            {
                "action": "tool_call",
                "name": "send_email",
                "args": {"to": "a@example.com", "subject": "s", "body": "b"},
            },
            {"action": "finish", "answer": "queued"},
        ],
    }
    engine = Engine(tmp_path / "home", model=RoutingModel(scripts), shop_db=SHOP_DB)
    state = engine.start("lead", allowed_tools=["send_email"])
    child = engine.store.load(state.active_child)
    assert state.status == child.status == "waiting_approval"
    assert "delegation_waiting" in event_types(engine, state.run_id)
    state = engine.decide(state.run_id, "approve")
    assert state.status == "done" and len(outbox(engine, state.run_id)) == 1
    assert engine.store.load(child.run_id).status == "done"
    engine.store.close()


def test_crash_resume_between_parent_and_child_creation(tmp_path):
    task = "worker calc"
    scripts = {
        "lead": [
            {"action": "delegate", "task": task, "tools": ["calc"]},
            {"action": "finish", "answer": "resumed"},
        ],
        task: [
            {"action": "tool_call", "name": "calc", "args": {"expression": "1+1"}},
            {"action": "finish", "answer": "2"},
        ],
    }
    engine = Engine(tmp_path / "home", model=RoutingModel(scripts), shop_db=SHOP_DB)
    run_id = "parent"
    workspace = tmp_path / "home" / "runs" / run_id / "workspace"
    workspace.mkdir(parents=True)
    engine.store.append(
        run_id,
        "started",
        {
            "goal": "lead",
            "script": [],
            "budgets": {},
            "workspace": str(workspace),
            "allowed_tools": ["calc"],
            "role": "lead",
            "parent_id": None,
        },
    )
    delegate = scripts["lead"][0]
    engine.store.append(run_id, "model_call", {"action": delegate, "tokens": 5})
    engine._handle(engine.store.load(run_id))
    child_id = engine.store.load(run_id).active_child
    assert child_id and child_id not in engine.store.run_ids()
    state = engine.resume(run_id)
    assert state.status == "done" and state.answer == "resumed"
    assert engine.store.load(child_id).status == "done"
    assert engine.store.rebuild(run_id).to_dict() == state.to_dict()
    engine.store.close()


def test_crash_resume_while_child_is_mid_run(tmp_path):
    task = "worker calc"
    scripts = {
        "lead": [
            {"action": "delegate", "task": task, "tools": ["calc"]},
            {"action": "finish", "answer": "resumed"},
        ],
        task: [
            {"action": "tool_call", "name": "calc", "args": {"expression": "1+1"}},
            {"action": "finish", "answer": "2"},
        ],
    }
    engine = Engine(tmp_path / "home", model=RoutingModel(scripts), shop_db=SHOP_DB)
    run_id = "parent"
    workspace = tmp_path / "home" / "runs" / run_id / "workspace"
    workspace.mkdir(parents=True)
    engine.store.append(
        run_id,
        "started",
        {
            "goal": "lead",
            "script": [],
            "budgets": {},
            "workspace": str(workspace),
            "allowed_tools": ["calc"],
            "role": "lead",
            "parent_id": None,
        },
    )
    engine.store.append(
        run_id, "model_call", {"action": scripts["lead"][0], "tokens": 5}
    )
    engine._handle(engine.store.load(run_id))
    parent = engine.store.load(run_id)
    started = next(
        event["data"]
        for event in engine.store.events(run_id)
        if event["type"] == "delegation_started"
    )
    engine.store.append(
        parent.active_child,
        "started",
        {
            "goal": task,
            "script": [],
            "budgets": started["budgets"],
            "workspace": str(workspace),
            "allowed_tools": ["calc"],
            "role": "worker",
            "parent_id": run_id,
        },
    )
    child_id = parent.active_child
    engine.store.append(
        child_id, "model_call", {"action": scripts[task][0], "tokens": 5}
    )
    engine._handle(engine.store.load(child_id))
    engine._execute(engine.store.load(child_id))
    assert engine.store.load(child_id).status == "running"

    state = engine.resume(run_id)
    assert state.status == "done" and state.answer == "resumed"
    child_events = event_types(engine, child_id)
    assert child_events.count("tool_result") == 1
    assert engine.store.load(child_id).status == "done"
    engine.store.close()


def test_cli_approval_reconstructs_persisted_ollama_model(
    tmp_path, monkeypatch, capsys
):
    from agent_orchestrator import model as model_module

    transport = FakeTransport(
        [
            response(
                {
                    "action": "tool_call",
                    "name": "send_email",
                    "args": {
                        "to": "a@example.com",
                        "subject": "Status",
                        "body": "Complete",
                    },
                }
            ),
            response({"answer": "sent"}),
        ]
    )
    monkeypatch.setattr(model_module, "OllamaTransport", lambda: transport)
    home = tmp_path / "home"
    base = ["--home", str(home), "--shop-db", str(SHOP_DB)]
    assert main([*base, "run", "--goal", "mail", "--model", "ollama"]) == 3
    run_id = capsys.readouterr().out.split()[1]
    assert main([*base, "approve", run_id, "--decision", "approve"]) == 0
    assert "answer: sent" in capsys.readouterr().out

    engine = Engine(home, shop_db=SHOP_DB)
    state = engine.store.load(run_id)
    assert state.model_config["type"] == "ollama" and state.status == "done"
    engine.store.close()


def test_suite_checkers_and_fake_eval_are_deterministic():
    tasks = load_suite()
    assert len(tasks) == 20 and isinstance(SuiteModel(tasks[0]), SuiteModel)
    first = run_eval("fake")
    second = run_eval("fake")
    assert first == second
    assert first["summary"]["successes"] == 20
    assert wilson(20, 20) == [0.8389, 1.0]


def test_infra_errors_are_excluded_from_eval_denominator():
    metrics = {
        "status": "done",
        "steps": 1,
        "tool_calls": 0,
        "tokens": 1,
        "repairs": 0,
        "approvals_requested": 0,
        "budget_hits": 0,
        "failures": [],
    }
    summary = summarize(
        "test",
        [
            {"id": "ok", "success": True, "excluded_infra": False, **metrics},
            {
                "id": "infra",
                "success": None,
                "excluded_infra": True,
                **metrics,
            },
        ],
    )
    assert summary["successes"] == summary["total"] == 1
    assert summary["suite_tasks"] == 2 and summary["excluded_infra"] == 1
