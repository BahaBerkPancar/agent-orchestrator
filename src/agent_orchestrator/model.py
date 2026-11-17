"""Model interface and the scripted fake model."""

import json
from dataclasses import dataclass, field
from typing import Protocol


class MalformedAction(Exception):
    def __init__(self, raw, message):
        super().__init__(message)
        self.raw = raw


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)

    def to_dict(self):
        return {"action": "tool_call", "name": self.name, "args": self.args}


@dataclass
class Delegate:
    """Stub for M2: hand a sub-task to a worker agent."""

    task: str
    tools: list = field(default_factory=list)

    def to_dict(self):
        return {"action": "delegate", "task": self.task, "tools": self.tools}


@dataclass
class Finish:
    answer: str

    def to_dict(self):
        return {"action": "finish", "answer": self.answer}


@dataclass
class Fail:
    reason: str

    def to_dict(self):
        return {"action": "fail", "reason": self.reason}


def parse_action(raw):
    """Validate a raw dict into an Action or raise MalformedAction."""
    if not isinstance(raw, dict):
        raise MalformedAction(raw, "action must be an object")
    kind = raw.get("action")
    if kind == "tool_call":
        if not isinstance(raw.get("name"), str) or not raw["name"]:
            raise MalformedAction(raw, "tool_call needs a string 'name'")
        args = raw.get("args", {})
        if not isinstance(args, dict):
            raise MalformedAction(raw, "tool_call 'args' must be an object")
        return ToolCall(raw["name"], args)
    if kind == "delegate":
        if not isinstance(raw.get("task"), str):
            raise MalformedAction(raw, "delegate needs a string 'task'")
        return Delegate(raw["task"], raw.get("tools", []))
    if kind == "finish":
        if not isinstance(raw.get("answer"), str):
            raise MalformedAction(raw, "finish needs a string 'answer'")
        return Finish(raw["answer"])
    if kind == "fail":
        if not isinstance(raw.get("reason"), str):
            raise MalformedAction(raw, "fail needs a string 'reason'")
        return Fail(raw["reason"])
    raise MalformedAction(raw, f"unknown action {kind!r}")


class Model(Protocol):
    def next(self, state): ...
    def count_tokens(self, raw: dict) -> int: ...


class ScriptedModel:
    """Replays a list of raw actions. The position is derived from the run state
    (number of model calls made), so a resumed process continues in the right place."""

    def __init__(self, script):
        self.script = script

    def next(self, state):
        i = state.step
        if i >= len(self.script):
            return Fail("script exhausted")
        return parse_action(self.script[i])

    def count_tokens(self, raw):
        return len(json.dumps(raw, sort_keys=True).split())
