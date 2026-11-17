import json
from pathlib import Path

import pytest

from agent_orchestrator.engine import Engine

SHOP_DB = Path(__file__).resolve().parents[1] / "examples" / "data" / "shop.db"


def tc(name, **args):
    return {"action": "tool_call", "name": name, "args": args}


def finish(answer="ok"):
    return {"action": "finish", "answer": answer}


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def _make(**kw):
        e = Engine(tmp_path / "home", shop_db=SHOP_DB, **kw)
        engines.append(e)
        return e

    yield _make
    for e in engines:
        e.store.close()


@pytest.fixture
def engine(make_engine):
    return make_engine()


def event_types(engine, run_id):
    return [e["type"] for e in engine.store.events(run_id)]


def outbox(engine, run_id):
    p = engine.home / "runs" / run_id / "workspace" / "outbox.jsonl"
    return (
        [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    )


@pytest.fixture
def engine_with_policy(make_engine):
    from agent_orchestrator.policy import Policy

    return make_engine(
        policy=Policy(
            [
                {
                    "tool": "send_email",
                    "match": {"to": "*@example.com"},
                    "decision": "approve",
                }
            ]
        )
    )
