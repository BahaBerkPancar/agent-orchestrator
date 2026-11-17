"""agent CLI. Exit codes: 0 done / ok, 1 failed, 2 usage error, 3 waiting_approval,
4 budget_exceeded, 5 cancelled, 6 unknown run, 7 run busy."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .engine import Engine, RunBusy, UsageError, load_script
from .policy import Policy

EXIT = {
    "done": 0,
    "failed": 1,
    "waiting_approval": 3,
    "budget_exceeded": 4,
    "cancelled": 5,
    "running": 0,
}
EXIT_USAGE, EXIT_UNKNOWN, EXIT_BUSY = 2, 6, 7
DEFAULT_SHOP_DB = Path(__file__).resolve().parents[2] / "examples" / "data" / "shop.db"


def _summary(ev):
    t, d = ev["type"], ev["data"]
    if t == "started":
        return f"goal: {d['goal']}"
    if t == "model_call":
        a = d["action"]
        detail = (
            a.get("name") or a.get("answer") or a.get("reason") or a.get("task") or ""
        )
        return f"{a.get('action', '?')} {detail} ({d['tokens']} tokens)".strip()
    if t == "tool_requested":
        return (
            f"{d['call_id']} {d['name']} {json.dumps(d['args'])[:110]} risk={d['risk']}"
        )
    if t == "approval_requested":
        return f"{d['call_id']} waiting for a human"
    if t == "approval_decided":
        extra = f" args={json.dumps(d['args'])}" if d.get("args") else ""
        return f"{d['call_id']} {d['decision']} by {d['by']}{extra}"
    if t == "tool_result":
        return f"{d['call_id']} {d['name']} ok={d['ok']}: {d['output'][:100]}"
    if t == "error":
        return d["message"]
    if t == "budget_hit":
        return d["reason"]
    if t == "finished":
        return f"{d['status']}: {d.get('answer') or d.get('reason') or ''}"
    return ""


def _report(state):
    print(f"run {state.run_id} status={state.status}")
    if state.status == "waiting_approval" and state.pending:
        p = state.pending
        print(f"  waiting: {p['name']} {json.dumps(p['args'])}")
        print(
            f"  approve with: agent approve {state.run_id} --decision approve|reject|edit [--args '{{...}}']"
        )
    if state.answer:
        print(f"  answer: {state.answer}")
    if state.reason:
        print(f"  reason: {state.reason}")
    return EXIT[state.status]


def build_parser():
    p = argparse.ArgumentParser(
        prog="agent",
        description="Durable, governed agent runs with a scripted fake model.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--home",
        default=os.environ.get("AGENT_HOME", ".agent"),
        help="state directory (db, workspaces, policy.toml); default $AGENT_HOME or ./.agent",
    )
    p.add_argument(
        "--shop-db", default=os.environ.get("AGENT_SHOP_DB", str(DEFAULT_SHOP_DB))
    )
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="start a run")
    r.add_argument("--goal", required=True)
    r.add_argument(
        "--script", required=True, help="JSON list of scripted model actions"
    )
    r.add_argument(
        "--policy",
        help="policy.toml, remembered for the whole run (default: <home>/policy.toml)",
    )
    r.add_argument("--max-steps", type=int)
    r.add_argument("--max-tool-calls", type=int)
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--max-wall-seconds", type=float)
    s = sub.add_parser("resume", help="continue an interrupted run")
    s.add_argument("run")
    a = sub.add_parser("approve", help="decide a pending approval")
    a.add_argument("run")
    a.add_argument("--decision", required=True, choices=["approve", "reject", "edit"])
    a.add_argument("--args", help="JSON object, required for edit")
    a.add_argument("--reason", default="")
    c = sub.add_parser("cancel", help="cancel a run")
    c.add_argument("run")
    sh = sub.add_parser("show", help="print a run's event timeline")
    sh.add_argument("run")
    sub.add_parser("runs", help="list runs")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return EXIT_USAGE
    delay = float(os.environ.get("AGENT_EVENT_DELAY", "0") or 0)
    engine = Engine(args.home, shop_db=args.shop_db, delay=delay)
    try:
        if args.cmd == "run":
            budgets = {
                k: v
                for k, v in {
                    "max_steps": args.max_steps,
                    "max_tool_calls": args.max_tool_calls,
                    "max_tokens": args.max_tokens,
                    "max_wall_seconds": args.max_wall_seconds,
                }.items()
                if v is not None
            }
            if args.policy and not Path(args.policy).is_file():
                raise UsageError(f"policy file not found: {args.policy}")
            script = load_script(args.script)
            Policy.load(args.policy)  # fail early on a malformed policy
            return _report(
                engine.start(args.goal, script, budgets, policy_path=args.policy)
            )
        if args.cmd == "runs":
            for rid in engine.store.run_ids():
                s = engine.store.load(rid)
                print(
                    f"{rid}  {s.status:<17} steps={s.step} tools={s.tool_calls}  {s.goal}"
                )
            return 0
        try:
            engine.store.load(args.run)
        except KeyError:
            print(f"unknown run: {args.run}", file=sys.stderr)
            return EXIT_UNKNOWN
        if args.cmd == "resume":
            return _report(engine.resume(args.run))
        if args.cmd == "cancel":
            return _report(engine.cancel(args.run))
        if args.cmd == "approve":
            try:
                extra = json.loads(args.args) if args.args else None
            except json.JSONDecodeError as e:
                raise UsageError(f"--args is not valid JSON: {e}") from None
            return _report(engine.decide(args.run, args.decision, extra, args.reason))
        if args.cmd == "show":
            st = engine.store.load(args.run)
            print(
                f"run {st.run_id}  status={st.status}  steps={st.step}  tool_calls={st.tool_calls}  tokens={st.tokens}"
            )
            print(f"goal: {st.goal}")
            for ev in engine.store.events(args.run):
                clock = time.strftime("%H:%M:%S", time.localtime(ev["ts"]))
                print(f"#{ev['seq']:<4} {clock}  {ev['type']:<18} {_summary(ev)}")
            return 0
    except RunBusy as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BUSY
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        engine.store.close()
    return 0
