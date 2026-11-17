"""Deterministic event-log replay and regression-test generation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .engine import Engine
from .model import MalformedAction
from .tools import default_registry


class _RecordedAction:
    def __init__(self, raw):
        self.raw = raw

    def to_dict(self):
        return self.raw


class ReplayModel:
    """Feed the exact model-call payloads recorded for each run."""

    def __init__(self, events):
        self.calls = {}
        for event in events:
            if event["type"] == "model_call":
                self.calls.setdefault(event["run_id"], []).append(event["data"])
        self.last_usage = {}

    def next(self, state):
        calls = self.calls.get(state.run_id, [])
        if state.step >= len(calls):
            self.last_usage = {"tokens": 0, "repairs": 0, "normalized": []}
            return _RecordedAction(
                {
                    "action": "fail",
                    "reason": f"recorded model output exhausted for run {state.run_id}",
                }
            )
        data = calls[state.step]
        self.last_usage = {
            "tokens": data["tokens"],
            "repairs": data.get("repairs", 0),
            "normalized": data.get("normalized", []),
        }
        if data.get("validation_error"):
            raise MalformedAction(data["action"], data["validation_error"])
        return _RecordedAction(data["action"])

    def count_tokens(self, raw):
        return self.last_usage["tokens"]


class ReplayPolicy:
    """Reproduce only decisions that the original log attributed to policy."""

    def __init__(self, events):
        pending = {}
        self.decisions = []
        for event in events:
            data = event["data"]
            key = (event["run_id"], data.get("call_id"))
            if event["type"] == "tool_requested":
                pending[key] = (data["name"], data["args"])
            elif event["type"] == "approval_decided" and data.get("by") == "policy":
                tool, args = pending[key]
                self.decisions.append((tool, args, data["decision"]))
        self.position = 0

    def decide(self, tool, args):
        if self.position >= len(self.decisions):
            return None
        recorded_tool, recorded_args, decision = self.decisions[self.position]
        if (tool, args) != (recorded_tool, recorded_args):
            return None
        self.position += 1
        return decision


@dataclass
class ReplayResult:
    identical: bool
    message: str
    original: dict | None
    replayed: dict | None
    state: object


def _tree_ids(store, root_id):
    states = {run_id: store.load(run_id) for run_id in store.run_ids()}
    selected = {root_id}
    changed = True
    while changed:
        changed = False
        for run_id, state in states.items():
            if state.parent_id in selected and run_id not in selected:
                selected.add(run_id)
                changed = True
    return selected


def recording_from_store(store, root_id):
    """Return a portable, JSON-serializable recording for a run tree."""
    root = store.load(root_id)
    run_ids = _tree_ids(store, root_id)
    source_events = sorted(
        [event for rid in run_ids for event in store.events(rid)],
        key=lambda event: event["seq"],
    )
    events = []
    for index, source in enumerate(source_events, 1):
        event = json.loads(json.dumps(source))
        event["seq"] = index
        event["ts"] = 0
        if "workspace" in event["data"]:
            event["data"]["workspace"] = "<recorded-workspace>"
        if "policy_path" in event["data"]:
            event["data"]["policy_path"] = None
        events.append(event)
    started = next(event["data"] for event in events if event["run_id"] == root_id)
    return {
        "version": 1,
        "root_id": root_id,
        "events": events,
        "workspace_seed": started.get("workspace_seed", {}),
        "expected": {"status": root.status, "answer": root.answer},
    }


def _restore_seed(workspace, seed):
    workspace.mkdir(parents=True, exist_ok=True)
    for name, encoded in seed.items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))


def _aliases(events, root_id):
    aliases = {root_id: "root"}
    child_counts = {}
    pending = list(events)
    while pending:
        progress = False
        for event in list(pending):
            if event["type"] != "started":
                pending.remove(event)
                continue
            parent = event["data"].get("parent_id")
            if event["run_id"] == root_id:
                pending.remove(event)
                progress = True
            elif parent in aliases:
                index = child_counts.get(parent, 0)
                child_counts[parent] = index + 1
                aliases[event["run_id"]] = f"{aliases[parent]}/{index}"
                pending.remove(event)
                progress = True
        if not progress:
            break
    return aliases


def _canonical(event, aliases):
    """Keep deterministic semantics; exclude timestamps, sequence numbers and IDs."""
    data = event["data"]
    kind = event["type"]
    if kind == "delegation_resumed":
        return None
    item = {"run": aliases.get(event["run_id"], "run"), "type": kind}
    if kind == "started":
        item.update(
            goal=data["goal"],
            budgets={
                key: value
                for key, value in data.get("budgets", {}).items()
                if key != "max_wall_seconds"
            },
            allowed_tools=data.get("allowed_tools"),
            role=data.get("role", "lead"),
        )
    elif kind == "model_call":
        item.update(
            action=data["action"],
            tokens=data["tokens"],
            repairs=data.get("repairs", 0),
            normalized=data.get("normalized", []),
            validation_error=data.get("validation_error"),
        )
    elif kind in {"tool_requested", "approval_requested"}:
        item.update(name=data.get("name"), args=data.get("args"))
        if kind == "tool_requested":
            item["risk"] = data["risk"]
    elif kind == "approval_decided":
        item.update(
            decision=data["decision"],
            by=data.get("by"),
            reason=data.get("reason", ""),
            args=data.get("args"),
        )
    elif kind == "tool_result":
        output = data["output"]
        if data["name"] == "send_email" and data["ok"] and (
            output == "email already in outbox (idempotent replay)"
            or (output.startswith("queued email to ") and output.endswith(" in outbox"))
        ):
            output = "send_email outbox write succeeded"
        item.update(
            name=data["name"],
            ok=data["ok"],
            result_hash=hashlib.sha256(output.encode()).hexdigest(),
        )
    elif kind == "delegation_started":
        item.update(
            task=data["task"],
            tools=data["tools"],
            budgets={
                key: value
                for key, value in data["budgets"].items()
                if key != "max_wall_seconds"
            },
        )
    elif kind == "delegation_waiting":
        pass
    elif kind == "delegation_finished":
        item.update(
            status=data["status"],
            answer=data.get("answer"),
            reason=data.get("reason"),
            steps=data["steps"],
            tool_calls=data["tool_calls"],
            tokens=data["tokens"],
            repairs=data.get("repairs", 0),
        )
    elif kind == "error":
        item["message"] = data["message"]
    elif kind == "budget_hit":
        item["reason"] = data["reason"]
    elif kind == "finished":
        item.update(
            status=data["status"],
            answer=data.get("answer"),
            reason=data.get("reason"),
        )
    return item


def compare_events(original, replayed, root_id):
    original_aliases = _aliases(original, root_id)
    replayed_aliases = _aliases(replayed, root_id)
    left = [
        item
        for event in original
        if (item := _canonical(event, original_aliases)) is not None
    ]
    right = [
        item
        for event in replayed
        if (item := _canonical(event, replayed_aliases)) is not None
    ]
    for index in range(max(len(left), len(right))):
        expected = left[index] if index < len(left) else None
        actual = right[index] if index < len(right) else None
        if expected != actual:
            return False, f"diverged at event {index + 1}", expected, actual
    return True, "identical", None, None


def _pending_leaf(engine, state):
    while state.active_child:
        state = engine.store.load(state.active_child)
    return state


def replay_recording(recording, destination, shop_db=None, registry=None):
    """Replay a portable recording in ``destination`` and compare event semantics."""
    destination = Path(destination)
    workspace = destination / "runs" / recording["root_id"] / "workspace"
    _restore_seed(workspace, recording.get("workspace_seed", {}))
    events = recording["events"]
    children = [
        event["data"]["child_id"]
        for event in events
        if event["type"] == "delegation_started"
    ]
    child_position = 0

    def next_child_id():
        nonlocal child_position
        if child_position < len(children):
            run_id = children[child_position]
        else:
            run_id = f"replay-extra-{child_position}"
        child_position += 1
        return run_id

    model = ReplayModel(events)
    engine = Engine(
        destination,
        model=model,
        registry=registry or default_registry(),
        policy=ReplayPolicy(events),
        shop_db=shop_db,
        run_id_factory=next_child_id,
    )
    root_id = recording["root_id"]
    started = next(
        event["data"]
        for event in events
        if event["run_id"] == root_id and event["type"] == "started"
    )
    try:
        state = engine.start(
            started["goal"],
            budgets=started.get("budgets"),
            run_id=root_id,
            allowed_tools=started.get("allowed_tools"),
            role=started.get("role", "lead"),
            parent_id=started.get("parent_id"),
            workspace=workspace,
            model_config=started.get("model_config"),
        )
        decisions = [
            event
            for event in events
            if event["type"] == "approval_decided"
            and event["data"].get("by") != "policy"
        ]
        position = 0
        while state.status == "waiting_approval" and position < len(decisions):
            leaf = _pending_leaf(engine, state)
            while position < len(decisions) and decisions[position]["run_id"] != leaf.run_id:
                position += 1
            if position >= len(decisions):
                break
            decision = decisions[position]["data"]
            state = engine.decide(
                root_id,
                decision["decision"],
                decision.get("args"),
                decision.get("reason", ""),
            )
            position += 1
        replayed = sorted(
            [
                event
                for rid in _tree_ids(engine.store, root_id)
                for event in engine.store.events(rid)
            ],
            key=lambda event: event["seq"],
        )
        identical, message, original_event, replayed_event = compare_events(
            events, replayed, root_id
        )
        return ReplayResult(
            identical, message, original_event, replayed_event, state
        )
    finally:
        engine.store.close()


def replay_run(store, run_id, shop_db=None, registry=None):
    recording = recording_from_store(store, run_id)
    with tempfile.TemporaryDirectory(prefix="agent-replay-") as temp:
        return replay_recording(recording, temp, shop_db=shop_db, registry=registry)


def write_regression_test(path, recording):
    path = Path(path)
    if path.suffix != ".py":
        raise ValueError("generated regression test path must end in .py")
    path.parent.mkdir(parents=True, exist_ok=True)
    fixture = path.with_suffix(".json")
    fixture.write_text(json.dumps(recording, indent=2, sort_keys=True) + "\n")
    test_name = "test_" + "".join(
        char if char.isalnum() else "_" for char in path.stem.removeprefix("test_")
    )
    path.write_text(
        "from pathlib import Path\n\n"
        "from agent_orchestrator.replay import assert_recorded_fixture\n\n\n"
        f"def {test_name}(tmp_path):\n"
        f"    fixture = Path(__file__).with_name({fixture.name!r})\n"
        "    assert_recorded_fixture(fixture, tmp_path)\n"
    )
    return fixture


def assert_recorded_fixture(path, temp_home, shop_db=None, registry=None):
    path = Path(path)
    recording = json.loads(path.read_text())
    if shop_db is None:
        configured = os.environ.get("AGENT_SHOP_DB")
        candidates = ([Path(configured)] if configured else []) + [
            parent / "examples" / "data" / "shop.db"
            for parent in [*path.resolve().parents, Path.cwd(), *Path(__file__).resolve().parents]
        ]
        shop_db = next((candidate for candidate in candidates if candidate.is_file()), None)
        if shop_db is None:
            raise FileNotFoundError(
                "demo shop database not found; pass shop_db=... or set AGENT_SHOP_DB"
            )
    result = replay_recording(
        recording, temp_home, shop_db=shop_db, registry=registry
    )
    assert result.identical, (
        f"{result.message}: expected {result.original!r}, got {result.replayed!r}"
    )
    assert result.state.status == recording["expected"]["status"]
    assert result.state.answer == recording["expected"]["answer"]


def copy_recording(recording):
    """Return a detached recording useful to callers that mutate it in tests."""
    return json.loads(json.dumps(recording))
