"""The run loop: model -> validate -> (approval) -> tool -> observe, checkpointed per event."""

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
    ):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.home / "agent.db")
        self.model = model
        self.registry = registry or default_registry()
        self.policy = (
            policy  # explicit override; otherwise loaded per run (see _policy_for)
        )
        self.shop_db = Path(shop_db) if shop_db else None
        self.clock = clock
        self.delay = (
            delay  # test hook: sleep after each event so a run can be killed mid-way
        )

    # -- helpers ---------------------------------------------------------
    def _emit(self, run_id, type_, data):
        ev = self.store.append(run_id, type_, data)
        if self.delay:
            time.sleep(self.delay)
        return ev

    def _model_for(self, state):
        return self.model or ScriptedModel(state.script)

    @contextlib.contextmanager
    def _locked(self, run_id):
        """Exclusive, non-blocking per-run lock held for a whole operation."""
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

    def _policy_for(self, st):
        if self.policy is not None:
            return self.policy
        return Policy.load(st.policy_path or self.home / "policy.toml")

    # -- public operations ----------------------------------------------
    def start(
        self, goal, script, budgets=None, run_id=None, policy_path=None
    ) -> RunState:
        run_id = run_id or uuid.uuid4().hex[:8]
        workspace = self.home / "runs" / run_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        with self._locked(run_id):
            self._emit(
                run_id,
                "started",
                {
                    "goal": goal,
                    "script": script,
                    "budgets": budgets or {},
                    "workspace": str(workspace),
                    "policy_path": str(Path(policy_path).resolve())
                    if policy_path
                    else None,
                },
            )
            return self._drive(run_id)

    def resume(self, run_id) -> RunState:
        with self._locked(run_id):
            return self._drive(run_id)

    def decide(self, run_id, decision, args=None, reason="") -> RunState:
        with self._locked(run_id):
            return self._decide(run_id, decision, args, reason)

    def _decide(self, run_id, decision, args, reason) -> RunState:
        state = self.store.load(run_id)
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
            if state.status in ("running", "waiting_approval"):
                self._emit(
                    run_id,
                    "finished",
                    {"status": "cancelled", "reason": "cancelled by user"},
                )
            return self.store.load(run_id)

    # -- the loop --------------------------------------------------------
    def _drive(self, run_id) -> RunState:
        t0 = self.clock()
        while True:
            st = self.store.load(run_id)
            if st.status != "running":
                return st
            if st.pending:
                if st.pending["approved"]:
                    self._execute(st)
                else:
                    self._gate(st)
                continue
            if st.pending_action is not None:
                self._handle(st)
                continue
            reason = self._budget_reason(st, t0)
            if reason:
                self._emit(run_id, "budget_hit", {"reason": reason})
                continue
            model = self._model_for(st)
            try:
                raw = model.next(st).to_dict()
            except MalformedAction as e:
                raw = e.raw
            self._emit(
                run_id, "model_call", {"action": raw, "tokens": model.count_tokens(raw)}
            )
            st = self.store.load(run_id)
            if st.tokens > st.budgets["max_tokens"]:
                self._emit(
                    run_id,
                    "budget_hit",
                    {"reason": f"max_tokens ({st.budgets['max_tokens']}) exceeded"},
                )

    def _budget_reason(self, st, t0):
        b = st.budgets
        if st.step >= b["max_steps"]:
            return f"max_steps ({b['max_steps']}) reached"
        if st.tokens >= b["max_tokens"]:
            return f"max_tokens ({b['max_tokens']}) reached"
        if self.clock() - t0 > b["max_wall_seconds"]:
            return f"max_wall_seconds ({b['max_wall_seconds']}) exceeded"
        return None

    def _error(self, st, message):
        self._emit(st.run_id, "error", {"message": message})
        if st.consecutive_errors + 1 >= MAX_CONSECUTIVE_ERRORS:
            self._emit(
                st.run_id,
                "finished",
                {
                    "status": "failed",
                    "reason": f"{MAX_CONSECUTIVE_ERRORS} invalid actions in a row",
                },
            )

    def _handle(self, st):
        run_id = st.run_id
        try:
            action = parse_action(st.pending_action)
        except MalformedAction as e:
            return self._error(st, f"malformed action: {e}")
        if isinstance(action, Finish):
            self._emit(run_id, "finished", {"status": "done", "answer": action.answer})
        elif isinstance(action, Fail):
            self._emit(
                run_id, "finished", {"status": "failed", "reason": action.reason}
            )
        elif isinstance(action, Delegate):
            self._error(st, "delegate is not implemented yet (M2)")
        elif isinstance(action, ToolCall):
            _, risk, errors = self.registry.check(action.name, action.args)
            if errors:
                return self._error(
                    st, f"invalid call to {action.name}: " + "; ".join(errors)
                )
            if risk == FORBIDDEN:
                return self._error(st, f"tool {action.name} is forbidden")
            if st.tool_calls >= st.budgets["max_tool_calls"]:
                return self._emit(
                    run_id,
                    "budget_hit",
                    {
                        "reason": f"max_tool_calls ({st.budgets['max_tool_calls']}) reached"
                    },
                )
            self._emit(
                run_id,
                "tool_requested",
                {
                    "call_id": f"c{st.tool_calls + 1}",
                    "name": action.name,
                    "args": action.args,
                    "risk": risk,
                },
            )

    def _gate(self, st):
        p = st.pending
        verdict = (
            self._policy_for(st).decide(p["name"], p["args"])
            if p["risk"] == REVIEW
            else None
        )
        if verdict:
            self._emit(
                st.run_id,
                "approval_decided",
                {
                    "call_id": p["call_id"],
                    "decision": verdict,
                    "by": "policy",
                    "reason": "policy.toml",
                },
            )
        else:
            self._emit(
                st.run_id,
                "approval_requested",
                {"call_id": p["call_id"], "name": p["name"], "args": p["args"]},
            )

    def _execute(self, st):
        p = st.pending
        tool = self.registry.get(p["name"])
        ctx = ToolContext(
            Path(st.workspace), f"{st.run_id}:{p['call_id']}", self.shop_db
        )
        ok = True
        try:
            output = tool.execute(p["args"], ctx)
        except ToolError as e:
            ok, output = False, f"tool error: {e}"
        except Exception as e:  # noqa: BLE001 - tool bugs must not kill the run
            ok, output = False, f"tool crashed: {type(e).__name__}: {e}"
        self._emit(
            st.run_id,
            "tool_result",
            {
                "call_id": p["call_id"],
                "name": p["name"],
                "ok": ok,
                "output": str(output)[:MAX_OUTPUT],
            },
        )


def load_script(path):
    try:
        script = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise UsageError(f"cannot read script {path}: {e}") from None
    if not isinstance(script, list):
        raise UsageError("script must be a JSON list of actions")
    return script


__all__ = ["Engine", "RunBusy", "UsageError", "load_script"]
