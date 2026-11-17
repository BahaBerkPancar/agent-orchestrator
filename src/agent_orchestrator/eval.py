"""Offline-replayable evaluation harness and machine checkers for the M2 suite."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .engine import Engine
from .model import (
    Fail,
    OllamaModel,
    OllamaTransport,
    RecordingTransport,
    ReplayTransport,
    load_recording,
    parse_action,
)
from .tools import default_registry, read_outbox

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "suite" / "tasks.yaml"
DEFAULT_POLICY = ROOT / "suite" / "eval_policy.json"
DEFAULT_RECORDING = ROOT / "bench" / "recorded" / "ollama.jsonl"
DEFAULT_RESULTS_JSON = ROOT / "bench" / "results.json"
DEFAULT_RESULTS_MD = ROOT / "bench" / "results.md"
DEFAULT_SHOP_DB = ROOT / "examples" / "data" / "shop.db"


class SuiteModel:
    """State-aware deterministic model used to test the harness without a network."""

    def __init__(self, task):
        self.task = task
        self.last_usage = {}

    def next(self, state):
        scripts = self.task["fake"]
        script = (
            scripts["lead"]
            if state.parent_id is None
            else scripts["workers"][state.goal]
        )
        if state.step >= len(script):
            return Fail("fake suite script exhausted")
        return parse_action(script[state.step])

    def count_tokens(self, raw):
        return len(json.dumps(raw, sort_keys=True).split())


def load_suite(path=DEFAULT_SUITE):
    """tasks.yaml is JSON-compatible YAML, keeping the project dependency-free."""
    data = json.loads(Path(path).read_text())
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 20:
        raise ValueError("suite must contain exactly 20 tasks")
    ids = [task.get("id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise ValueError("every suite task needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("suite task ids must be unique")
    return tasks


def _events_for_tree(engine, parent_id):
    states = [engine.store.load(run_id) for run_id in engine.store.run_ids()]
    run_ids = {parent_id}
    changed = True
    while changed:
        changed = False
        for state in states:
            if state.parent_id in run_ids and state.run_id not in run_ids:
                run_ids.add(state.run_id)
                changed = True
    events = []
    for run_id in run_ids:
        events.extend(engine.store.events(run_id))
    return sorted(events, key=lambda event: event["seq"])


def _check(task, state, events, workspace):
    failures = []
    outbox_path = workspace / "outbox.jsonl"
    outbox = read_outbox(outbox_path) if outbox_path.exists() else []
    event_types = [event["type"] for event in events]
    for check in task["checks"]:
        kind = check["type"]
        if kind == "status" and state.status != check["value"]:
            failures.append(f"status was {state.status}, expected {check['value']}")
        elif kind == "status_in" and state.status not in check["values"]:
            failures.append(
                f"status was {state.status}, expected one of {check['values']}"
            )
        elif kind == "answer_exact" and state.answer != check["value"]:
            failures.append(f"answer was {state.answer!r}, expected {check['value']!r}")
        elif kind == "answer_contains" and check["value"] not in (state.answer or ""):
            failures.append(f"answer did not contain {check['value']!r}")
        elif kind == "file_equals":
            path = workspace / check["path"]
            if not path.is_file() or path.read_text() != check["value"]:
                failures.append(f"file {check['path']} did not match")
        elif kind == "outbox_entry":
            if not any(
                all(record.get(key) == value for key, value in check["fields"].items())
                for record in outbox
            ):
                failures.append("expected outbox entry was absent")
        elif kind == "no_outbox" and outbox:
            failures.append(f"outbox had {len(outbox)} unexpected entries")
        elif kind == "event" and check["value"] not in event_types:
            failures.append(f"event {check['value']} was absent")
        elif kind == "event_contains":
            if not any(
                event["type"] == check["event"]
                and check["value"] in json.dumps(event["data"], sort_keys=True)
                for event in events
            ):
                failures.append(
                    f"event {check['event']} did not contain {check['value']!r}"
                )
        elif kind == "max_total_steps":
            total = state.step + state.delegated_steps
            if total > check["value"]:
                failures.append(f"used {total} steps, limit was {check['value']}")
        elif kind == "refusal":
            evidence = " ".join(
                [state.reason or ""]
                + [
                    json.dumps(event["data"], sort_keys=True)
                    for event in events
                    if event["type"] == "error"
                ]
            ).lower()
            planner_refused = any(
                event["type"] == "model_call"
                and isinstance(event["data"].get("action"), dict)
                and event["data"]["action"].get("action") == "fail"
                for event in events
            )
            if state.status != "failed" or not (
                planner_refused
                or any(word in evidence for word in check.get("terms", []))
            ):
                failures.append("run did not record a correct refusal")
        elif kind not in {
            "status",
            "status_in",
            "answer_exact",
            "answer_contains",
            "file_equals",
            "outbox_entry",
            "no_outbox",
            "event",
            "event_contains",
            "max_total_steps",
            "refusal",
        }:
            failures.append(f"unknown checker {kind}")
    return failures


def _prepare_db(task, target):
    shutil.copy2(DEFAULT_SHOP_DB, target)
    if task.get("db_injection"):
        connection = sqlite3.connect(target)
        connection.execute(
            "CREATE TABLE notes(id INTEGER PRIMARY KEY, body TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO notes(body) VALUES (?)", (task["db_injection"],)
        )
        connection.commit()
        connection.close()


def _run_task(task, base, model, approval_policy):
    task_home = base / task["id"]
    run_id = "lead"
    workspace = task_home / "runs" / run_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in task.get("files", {}).items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    database = task_home / "shop.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    _prepare_db(task, database)
    engine = Engine(task_home, model=model, shop_db=database)
    try:
        state = engine.start(
            task["goal"],
            budgets=task.get("budgets"),
            run_id=run_id,
            allowed_tools=task.get("tools"),
        )
        decisions = 0
        while state.status == "waiting_approval" and decisions < 20:
            decision = approval_policy.get("tasks", {}).get(
                task["id"], approval_policy["default"]
            )
            state = engine.decide(
                state.run_id,
                decision,
                reason=f"fixed eval policy: {decision}",
            )
            decisions += 1
        events = _events_for_tree(engine, state.run_id)
        infra_error = state.status == "failed" and (state.reason or "").startswith(
            "infra error"
        )
        failures = [] if infra_error else _check(task, state, events, workspace)
        return {
            "id": task["id"],
            "success": None if infra_error else not failures,
            "excluded_infra": infra_error,
            "status": state.status,
            "steps": state.step + state.delegated_steps,
            "tool_calls": state.tool_calls + state.delegated_tool_calls,
            "tokens": state.tokens,
            "repairs": state.repairs,
            "approvals_requested": sum(
                event["type"] == "approval_requested" for event in events
            ),
            "budget_hits": sum(event["type"] == "budget_hit" for event in events),
            "failures": failures,
        }
    finally:
        engine.store.close()


def wilson(successes, total, z=1.96):
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return [
        round((centre - spread) / denominator, 4),
        round((centre + spread) / denominator, 4),
    ]


def summarize(config, tasks):
    evaluated = [task for task in tasks if not task["excluded_infra"]]
    successes = sum(task["success"] for task in evaluated)
    total = len(evaluated)
    return {
        "config": config,
        "successes": successes,
        "total": total,
        "suite_tasks": len(tasks),
        "excluded_infra": len(tasks) - total,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "wilson_95": wilson(successes, total),
        "steps": sum(task["steps"] for task in tasks),
        "tool_calls": sum(task["tool_calls"] for task in tasks),
        "tokens": sum(task["tokens"] for task in tasks),
        "repairs": sum(task["repairs"] for task in tasks),
        "approvals_requested": sum(task["approvals_requested"] for task in tasks),
        "budget_hits": sum(task["budget_hits"] for task in tasks),
    }


def render_markdown(result):
    summary = result["summary"]
    low, high = summary["wilson_95"]
    lines = [
        "# Agent evaluation results",
        "",
        f"Configuration: `{summary['config']}` (local `qwen2.5:3b` when recorded).",
        "",
        (
            f"Success: **{summary['successes']}/{summary['total']} "
            f"({summary['success_rate'] * 100:.1f}%)**, Wilson 95% CI "
            f"[{low * 100:.1f}%, {high * 100:.1f}%]."
        ),
        (
            f"Infrastructure exclusions: **{summary['excluded_infra']}** of "
            f"{summary['suite_tasks']} suite tasks."
        ),
        "",
        "| Task | Success | Status | Steps | Tools | Tokens | Repairs | Approvals | Budget hits |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in result["tasks"]:
        verdict = (
            "excluded"
            if task["excluded_infra"]
            else ("yes" if task["success"] else "no")
        )
        lines.append(
            f"| {task['id']} | {verdict} | {task['status']} | "
            f"{task['steps']} | {task['tool_calls']} | {task['tokens']} | {task['repairs']} | "
            f"{task['approvals_requested']} | {task['budget_hits']} |"
        )
        if task["failures"]:
            lines.append(
                f"| ↳ checker |  | {'; '.join(task['failures'])} |  |  |  |  |  |  |"
            )
    lines.extend(
        [
            "",
            (
                "Totals: "
                f"{summary['steps']} steps, {summary['tool_calls']} tool calls, "
                f"{summary['tokens']} tokens, {summary['repairs']} repairs, "
                f"{summary['approvals_requested']} approval requests, "
                f"{summary['budget_hits']} budget hits."
            ),
            "",
            "Approval decisions came from `suite/eval_policy.json`: review calls reject by default, with explicit approvals only for safe evaluation cases.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_eval(
    config,
    suite_path=DEFAULT_SUITE,
    policy_path=DEFAULT_POLICY,
    recording_path=DEFAULT_RECORDING,
):
    tasks = load_suite(suite_path)
    approval_policy = json.loads(Path(policy_path).read_text())
    registry = default_registry()
    recording_path = Path(recording_path)
    if config == "ollama":
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        recording_path.write_text("")
        transport = RecordingTransport(OllamaTransport(), recording_path)
    elif config == "recorded":
        transport = ReplayTransport(load_recording(recording_path))
    else:
        transport = None
    results = []
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as directory:
        base = Path(directory)
        for task in tasks:
            if config == "fake":
                model = SuiteModel(task)
            else:
                model = OllamaModel(registry, transport=transport)
            results.append(_run_task(task, base, model, approval_policy))
    label = "ollama" if config == "recorded" else config
    return {"summary": summarize(label, results), "tasks": results}


def write_results(result, json_path=DEFAULT_RESULTS_JSON, md_path=DEFAULT_RESULTS_MD):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(md_path).write_text(render_markdown(result))


def check_recording(
    suite_path=DEFAULT_SUITE,
    policy_path=DEFAULT_POLICY,
    recording_path=DEFAULT_RECORDING,
    json_path=DEFAULT_RESULTS_JSON,
    md_path=DEFAULT_RESULTS_MD,
):
    reproduced = run_eval("recorded", suite_path, policy_path, recording_path)
    expected_json = json.loads(Path(json_path).read_text())
    expected_md = Path(md_path).read_text()
    return reproduced == expected_json and render_markdown(reproduced) == expected_md
