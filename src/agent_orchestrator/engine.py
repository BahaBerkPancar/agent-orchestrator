"""Durable model/tool run loop, including governed lead-to-worker delegation."""

import base64
import contextlib
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from .model import (
    Delegate,
    Fail,
    Finish,
    MalformedAction,
    ModelTransportError,
    OllamaModel,
    ScriptedModel,
    ToolCall,
    parse_action,
)
from .policy import Policy
from .state import RunState
from .store import Store
from .tools import (
    FORBIDDEN,
    MAX_OUTPUT,
    REVIEW,
    Registry,
    ToolContext,
    ToolError,
    default_registry,
)

MAX_CONSECUTIVE_ERRORS = 3


class UsageError(Exception):
    pass


class RunBusy(Exception):
    """Another process is driving this run."""


class Engine:
    def __init__(
        self,
        home,
        model=None,
        registry: Registry | None = None,
        policy: Policy | None = None,
        shop_db=None,
        clock=time.monotonic,
        delay=0.0,
        run_id_factory=None,
    ):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.home / "agent.db")
        self.model = model
        self.registry = registry or default_registry()
        self.policy = policy
        self.shop_db = Path(shop_db) if shop_db else None
        self.clock = clock
        self.delay = delay
        self.run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex[:8])

    @staticmethod
    def _workspace_seed(workspace):
        """Capture the files present before a run so replay can seed a fresh sandbox."""
        seed = {}
        for path in sorted(workspace.rglob("*")):
            if path.is_file() and not path.is_symlink():
                seed[path.relative_to(workspace).as_posix()] = base64.b64encode(
                    path.read_bytes()
                ).decode("ascii")
        return seed

    def _emit(self, run_id, type_, data):
        event = self.store.append(run_id, type_, data)
        if self.delay:
            time.sleep(self.delay)
        return event

    def _model_for(self, state):
        if self.model is not None:
            return self.model
        if state.model_config.get("type") == "ollama":
            return OllamaModel(
                self.registry,
                model=state.model_config.get("model", "qwen2.5:3b"),
                history_chars=state.model_config.get("history_chars", 6000),
            )
        return ScriptedModel(state.script)

    def _allowed(self, state):
        if state.allowed_tools is None:
            return set(self.registry.tools)
        return set(state.allowed_tools)

    @contextlib.contextmanager
    def _locked(self, run_id):
        lock_dir = self.home / "locks"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(
            lock_dir / f"{run_id}.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunBusy(
                    f"run {run_id} is busy (another process is driving it)"
                ) from None
            yield
        finally:
            os.close(fd)

    def _policy_for(self, state):
        if self.policy is not None:
            return self.policy
        return Policy.load(state.policy_path or self.home / "policy.toml")

    def start(
        self,
        goal,
        script=None,
        budgets=None,
        run_id=None,
        policy_path=None,
        allowed_tools=None,
        role="lead",
        parent_id=None,
        workspace=None,
        model_config=None,
    ) -> RunState:
        run_id = run_id or self.run_id_factory()
        workspace = Path(workspace or self.home / "runs" / run_id / "workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_seed = self._workspace_seed(workspace)
        if allowed_tools is None:
            allowed_tools = list(self.registry.tools)
        if model_config is None:
            if self.model is not None and hasattr(self.model, "configuration"):
                model_config = self.model.configuration()
            else:
                model_config = {"type": "scripted"}
        with self._locked(run_id):
            self._emit(
                run_id,
                "started",
                {
                    "goal": goal,
                    "script": script or [],
                    "budgets": budgets or {},
                    "workspace": str(workspace),
                    "workspace_seed": workspace_seed,
                    "policy_path": str(Path(policy_path).resolve())
                    if policy_path
                    else None,
                    "allowed_tools": list(allowed_tools),
                    "model_config": model_config,
                    "role": role,
                    "parent_id": parent_id,
                },
            )
            return self._drive(run_id)

    def resume(self, run_id) -> RunState:
        with self._locked(run_id):
            state = self.store.load(run_id)
            if state.status == "waiting_approval" and state.active_child:
                child = self.store.load(state.active_child)
                if child.status != "waiting_approval":
                    self._emit(run_id, "delegation_resumed", {"child_id": child.run_id})
            return self._drive(run_id)

    def decide(self, run_id, decision, args=None, reason="") -> RunState:
        with self._locked(run_id):
            return self._decide(run_id, decision, args, reason)

    def _decide(self, run_id, decision, args, reason) -> RunState:
        state = self.store.load(run_id)
        if state.active_child:
            child = self.store.load(state.active_child)
            if state.status != "waiting_approval" or child.status != "waiting_approval":
                raise UsageError(
                    f"run {run_id} is not waiting for approval (status {state.status})"
                )
            with self._locked(child.run_id):
                self._decide(child.run_id, decision, args, reason)
            self._emit(run_id, "delegation_resumed", {"child_id": child.run_id})
            return self._drive(run_id)
        if state.status != "waiting_approval" or not state.pending:
            raise UsageError(
                f"run {run_id} is not waiting for approval (status {state.status})"
            )
        if decision == "edit":
            if not isinstance(args, dict):
                raise UsageError("--decision edit needs --args '{...}'")
            _, risk, errors = self.registry.check(state.pending["name"], args)
            if errors:
                raise UsageError("edited args rejected: " + "; ".join(errors))
            if risk == FORBIDDEN:
                raise UsageError("edited call is forbidden")
        data = {
            "call_id": state.pending["call_id"],
            "decision": decision,
            "by": "human",
            "reason": reason,
        }
        if decision == "edit":
            data["args"] = args
        self._emit(run_id, "approval_decided", data)
        return self._drive(run_id)

    def cancel(self, run_id) -> RunState:
        with self._locked(run_id):
            state = self.store.load(run_id)
            if state.active_child:
                child = self.store.load(state.active_child)
                if child.status in ("running", "waiting_approval"):
                    with self._locked(child.run_id):
                        self._emit(
                            child.run_id,
                            "finished",
                            {"status": "cancelled", "reason": "parent cancelled"},
                        )
            if state.status in ("running", "waiting_approval"):
                self._emit(
                    run_id,
                    "finished",
                    {"status": "cancelled", "reason": "cancelled by user"},
                )
            return self.store.load(run_id)

    def _drive(self, run_id, deadline=None) -> RunState:
        if deadline is None:
            initial_state = self.store.load(run_id)
            deadline = self.clock() + initial_state.budgets["max_wall_seconds"]
        while True:
            state = self.store.load(run_id)
            if state.status != "running":
                return state
            if state.active_child:
                self._continue_child(state, deadline)
                continue
            if state.pending:
                if state.pending["approved"]:
                    self._execute(state)
                else:
                    self._gate(state)
                continue
            if state.pending_action is not None:
                self._handle(state, deadline)
                continue
            reason = self._budget_reason(state, deadline)
            if reason:
                self._emit(run_id, "budget_hit", {"reason": reason})
                continue
            model = self._model_for(state)
            validation_error = None
            try:
                raw = model.next(state).to_dict()
            except MalformedAction as exc:
                raw = exc.raw
                validation_error = str(exc)
            except ModelTransportError as exc:
                self._emit(
                    run_id,
                    "finished",
                    {"status": "failed", "reason": str(exc)},
                )
                continue
            usage = getattr(model, "last_usage", {})
            tokens = usage.get("tokens", model.count_tokens(raw))
            self._emit(
                run_id,
                "model_call",
                {
                    "action": raw,
                    "tokens": tokens,
                    "repairs": usage.get("repairs", 0),
                    "normalized": usage.get("normalized", []),
                    "validation_error": validation_error,
                },
            )
            state = self.store.load(run_id)
            if state.tokens > state.budgets["max_tokens"]:
                self._emit(
                    run_id,
                    "budget_hit",
                    {"reason": f"max_tokens ({state.budgets['max_tokens']}) exceeded"},
                )

    def _budget_reason(self, state, deadline):
        budgets = state.budgets
        if state.step + state.delegated_steps >= budgets["max_steps"]:
            return f"max_steps ({budgets['max_steps']}) reached"
        if state.tokens >= budgets["max_tokens"]:
            return f"max_tokens ({budgets['max_tokens']}) reached"
        if self.clock() > deadline:
            return f"max_wall_seconds ({budgets['max_wall_seconds']}) exceeded"
        return None

    def _error(self, state, message):
        self._emit(state.run_id, "error", {"message": message})
        if state.consecutive_errors + 1 >= MAX_CONSECUTIVE_ERRORS:
            self._emit(
                state.run_id,
                "finished",
                {
                    "status": "failed",
                    "reason": f"{MAX_CONSECUTIVE_ERRORS} invalid actions in a row",
                },
            )

    def _handle(self, state, deadline=None):
        run_id = state.run_id
        if state.pending_action_error:
            return self._error(state, state.pending_action_error)
        try:
            action = parse_action(state.pending_action)
        except MalformedAction as exc:
            return self._error(state, f"malformed action: {exc}")
        if isinstance(action, Finish):
            self._emit(run_id, "finished", {"status": "done", "answer": action.answer})
        elif isinstance(action, Fail):
            self._emit(
                run_id, "finished", {"status": "failed", "reason": action.reason}
            )
        elif isinstance(action, Delegate):
            self._delegate(state, action, deadline)
        elif isinstance(action, ToolCall):
            if action.name not in self._allowed(state):
                return self._error(
                    state, f"tool {action.name} is outside this run's allowed toolset"
                )
            _, risk, errors = self.registry.check(action.name, action.args)
            if errors:
                return self._error(
                    state, f"invalid call to {action.name}: " + "; ".join(errors)
                )
            if risk == FORBIDDEN:
                return self._error(state, f"tool {action.name} is forbidden")
            used = state.tool_calls + state.delegated_tool_calls
            if used >= state.budgets["max_tool_calls"]:
                return self._emit(
                    run_id,
                    "budget_hit",
                    {
                        "reason": f"max_tool_calls ({state.budgets['max_tool_calls']}) reached"
                    },
                )
            self._emit(
                run_id,
                "tool_requested",
                {
                    "call_id": f"c{state.tool_calls + 1}",
                    "name": action.name,
                    "args": action.args,
                    "risk": risk,
                },
            )

    def _delegate(self, state, action, deadline=None):
        if state.role != "lead":
            return self._error(state, "worker runs cannot delegate")
        if not action.tools:
            return self._error(state, "M2 delegate needs at least one worker tool")
        parent_tools = self._allowed(state)
        if not set(action.tools).issubset(parent_tools):
            return self._error(
                state, "delegated tools must be a subset of the parent's tools"
            )
        for name in action.tools:
            tool = self.registry.get(name)
            if tool is None:
                return self._error(state, f"cannot delegate unknown tool {name}")
            if tool.risk == FORBIDDEN:
                return self._error(state, f"cannot delegate forbidden tool {name}")
        wall_remaining = state.budgets["max_wall_seconds"]
        if deadline is not None:
            wall_remaining = max(0, deadline - self.clock())
        remaining = {
            "max_steps": state.budgets["max_steps"]
            - state.step
            - state.delegated_steps,
            "max_tool_calls": state.budgets["max_tool_calls"]
            - state.tool_calls
            - state.delegated_tool_calls,
            "max_tokens": state.budgets["max_tokens"] - state.tokens,
            "max_wall_seconds": wall_remaining,
        }
        if any(remaining[key] <= 0 for key in remaining):
            return self._emit(
                state.run_id,
                "budget_hit",
                {"reason": "no parent budget remains for delegation"},
            )
        child_budgets = {
            key: min(action.budgets.get(key, value), value)
            for key, value in remaining.items()
        }
        child_id = self.run_id_factory()
        self._emit(
            state.run_id,
            "delegation_started",
            {
                "child_id": child_id,
                "task": action.task,
                "tools": action.tools,
                "budgets": child_budgets,
                "workspace": state.workspace,
                "policy_path": state.policy_path,
            },
        )

    def _continue_child(self, parent, deadline=None):
        child_id = parent.active_child
        try:
            child = self.store.load(child_id)
        except KeyError:
            event = next(
                event
                for event in reversed(self.store.events(parent.run_id))
                if event["type"] == "delegation_started"
                and event["data"]["child_id"] == child_id
            )
            data = event["data"]
            self._emit(
                child_id,
                "started",
                {
                    "goal": data["task"],
                    "script": [],
                    "budgets": data["budgets"],
                    "workspace": data["workspace"],
                    "policy_path": data.get("policy_path"),
                    "model_config": parent.model_config,
                    "allowed_tools": data["tools"],
                    "role": "worker",
                    "parent_id": parent.run_id,
                },
            )
            child = self.store.load(child_id)
        if child.status == "running":
            with self._locked(child_id):
                child = self._drive(child_id)
        if child.status == "waiting_approval":
            self._emit(
                parent.run_id,
                "delegation_waiting",
                {"child_id": child_id, "call_id": child.pending["call_id"]},
            )
            return
        self._emit(
            parent.run_id,
            "delegation_finished",
            {
                "child_id": child_id,
                "status": child.status,
                "answer": child.answer,
                "reason": child.reason,
                "steps": child.step + child.delegated_steps,
                "tool_calls": child.tool_calls + child.delegated_tool_calls,
                "tokens": child.tokens,
                "repairs": child.repairs,
            },
        )

    def _gate(self, state):
        pending = state.pending
        verdict = (
            self._policy_for(state).decide(pending["name"], pending["args"])
            if pending["risk"] == REVIEW
            else None
        )
        if verdict:
            self._emit(
                state.run_id,
                "approval_decided",
                {
                    "call_id": pending["call_id"],
                    "decision": verdict,
                    "by": "policy",
                    "reason": "policy.toml",
                },
            )
        else:
            self._emit(
                state.run_id,
                "approval_requested",
                {
                    "call_id": pending["call_id"],
                    "name": pending["name"],
                    "args": pending["args"],
                },
            )

    def _execute(self, state):
        pending = state.pending
        tool = self.registry.get(pending["name"])
        context = ToolContext(
            Path(state.workspace),
            f"{state.run_id}:{pending['call_id']}",
            self.shop_db,
        )
        ok = True
        try:
            output = tool.execute(pending["args"], context)
        except ToolError as exc:
            ok, output = False, f"tool error: {exc}"
        except Exception as exc:  # noqa: BLE001
            ok, output = False, f"tool crashed: {type(exc).__name__}: {exc}"
        self._emit(
            state.run_id,
            "tool_result",
            {
                "call_id": pending["call_id"],
                "name": pending["name"],
                "ok": ok,
                "output": str(output)[:MAX_OUTPUT],
            },
        )


def load_script(path):
    try:
        script = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UsageError(f"cannot read script {path}: {exc}") from None
    if not isinstance(script, list):
        raise UsageError("script must be a JSON list of actions")
    return script


__all__ = ["Engine", "RunBusy", "UsageError", "load_script"]
