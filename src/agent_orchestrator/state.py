"""Run state and the pure reducer that derives it from the event log."""

import copy
from dataclasses import asdict, dataclass, field

STATUSES = (
    "running",
    "waiting_approval",
    "done",
    "failed",
    "budget_exceeded",
    "cancelled",
)
EVENT_TYPES = (
    "started",
    "model_call",
    "tool_requested",
    "approval_requested",
    "approval_decided",
    "tool_result",
    "error",
    "delegation_started",
    "delegation_waiting",
    "delegation_resumed",
    "delegation_finished",
    "budget_hit",
    "finished",
)
DEFAULT_BUDGETS = {
    "max_steps": 20,
    "max_tool_calls": 15,
    "max_tokens": 5000,
    "max_wall_seconds": 300,
}


@dataclass
class RunState:
    run_id: str
    goal: str = ""
    status: str = "running"
    messages: list = field(default_factory=list)
    scratchpad: dict = field(default_factory=dict)
    step: int = 0  # model calls so far
    tool_calls: int = 0
    tokens: int = 0
    repairs: int = 0
    budgets: dict = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    script: list = field(default_factory=list)
    workspace: str = ""
    policy_path: str | None = None
    model_config: dict = field(default_factory=lambda: {"type": "scripted"})
    role: str = "lead"
    parent_id: str | None = None
    allowed_tools: list[str] | None = None
    active_child: str | None = None
    delegated_steps: int = 0
    delegated_tool_calls: int = 0
    pending: dict | None = (
        None  # tool call waiting to run: call_id, name, args, risk, approved
    )
    pending_action: dict | None = None  # model output not yet turned into an event
    pending_action_error: str | None = None
    consecutive_errors: int = 0
    answer: str | None = None
    reason: str | None = None
    last_seq: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def initial(run_id):
    return RunState(run_id=run_id)


def reduce(state: RunState, event: dict) -> RunState:
    """Pure: (state, event) -> new state. event = {seq, type, data}."""
    s = copy.deepcopy(state)
    t, d = event["type"], event["data"]
    s.last_seq = event["seq"]
    if t == "started":
        s.goal = d["goal"]
        s.budgets = {**DEFAULT_BUDGETS, **d.get("budgets", {})}
        s.script = d.get("script", [])
        s.workspace = d.get("workspace", "")
        s.policy_path = d.get("policy_path")
        s.model_config = d.get("model_config", {"type": "scripted"})
        s.role = d.get("role", "lead")
        s.parent_id = d.get("parent_id")
        s.allowed_tools = d.get("allowed_tools")
        s.status = "running"
        s.messages.append({"role": "user", "content": d["goal"]})
    elif t == "model_call":
        s.step += 1
        s.tokens += d["tokens"]
        s.repairs += d.get("repairs", 0)
        s.pending_action = d["action"]
        s.pending_action_error = d.get("validation_error")
        s.messages.append({"role": "assistant", "content": d["action"]})
    elif t == "tool_requested":
        s.tool_calls += 1
        s.consecutive_errors = 0
        s.pending_action = None
        s.pending_action_error = None
        s.pending = {
            "call_id": d["call_id"],
            "name": d["name"],
            "args": d["args"],
            "risk": d["risk"],
            "approved": d["risk"] == "safe",
        }
    elif t == "approval_requested":
        s.status = "waiting_approval"
    elif t == "approval_decided":
        s.status = "running"
        if d["decision"] == "reject":
            s.messages.append(
                {
                    "role": "tool",
                    "name": s.pending["name"],
                    "content": "REJECTED by approver: " + d.get("reason", ""),
                }
            )
            s.pending = None
        else:
            s.pending["approved"] = True
            if d["decision"] == "edit":
                s.pending["args"] = d["args"]
    elif t == "tool_result":
        s.messages.append({"role": "tool", "name": d["name"], "content": d["output"]})
        s.scratchpad["last_tool"] = d["name"]
        s.scratchpad["last_result"] = d["output"]
        s.pending = None
    elif t == "error":
        s.pending_action = None
        s.pending_action_error = None
        s.consecutive_errors += 1
        s.messages.append({"role": "system", "content": "ERROR: " + d["message"]})
    elif t == "delegation_started":
        s.pending_action = None
        s.pending_action_error = None
        s.consecutive_errors = 0
        s.active_child = d["child_id"]
    elif t == "delegation_waiting":
        s.status = "waiting_approval"
    elif t == "delegation_resumed":
        s.status = "running"
    elif t == "delegation_finished":
        s.status = "running"
        s.active_child = None
        s.delegated_steps += d["steps"]
        s.delegated_tool_calls += d["tool_calls"]
        s.tokens += d["tokens"]
        s.repairs += d.get("repairs", 0)
        outcome = d.get("answer") or d.get("reason") or d["status"]
        s.messages.append(
            {
                "role": "tool",
                "name": "delegate",
                "content": f"worker {d['status']}: {outcome}",
            }
        )
    elif t == "budget_hit":
        s.pending_action = None
        s.pending_action_error = None
        s.pending = None
        s.status = "budget_exceeded"
        s.reason = d["reason"]
    elif t == "finished":
        s.pending_action = None
        s.pending_action_error = None
        s.status = d["status"]
        s.answer = d.get("answer")
        s.reason = d.get("reason")
    else:
        raise ValueError(f"unknown event type {t}")
    return s


def rebuild(run_id, events) -> RunState:
    s = initial(run_id)
    for e in events:
        s = reduce(s, e)
    return s
