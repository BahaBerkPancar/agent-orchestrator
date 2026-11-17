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
from .model import OllamaModel
from .policy import Policy
from .tools import default_registry

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
        normalized = (
            f" normalized={'; '.join(d['normalized'])}" if d.get("normalized") else ""
        )
        return f"{a.get('action', '?')} {detail} ({d['tokens']} tokens){normalized}".strip()
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
    if t == "delegation_started":
        return f"worker {d['child_id']} task={d['task']} tools={','.join(d['tools'])}"
    if t == "delegation_waiting":
        return f"worker {d['child_id']} waiting for approval"
    if t == "delegation_resumed":
        return f"worker {d['child_id']} approval resolved"
    if t == "delegation_finished":
        return f"worker {d['child_id']} {d['status']}"
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
    elif state.status == "waiting_approval" and state.active_child:
        print(f"  waiting: approval requested by worker {state.active_child}")
    if state.answer:
        print(f"  answer: {state.answer}")
    if state.reason:
        print(f"  reason: {state.reason}")
    return EXIT[state.status]


def build_parser():
    p = argparse.ArgumentParser(
        prog="agent",
        description="Durable, governed agent runs with scripted or local Ollama planners.",
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
    r.add_argument("--model", choices=["scripted", "ollama"], default="scripted")
    r.add_argument("--script", help="JSON list of actions (required for scripted)")
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
    rp = sub.add_parser("replay", help="re-execute a run from recorded model outputs")
    rp.add_argument("run")
    rp.add_argument("--to-test", help="write a pytest plus adjacent JSON fixture")
    rp.add_argument("--home", dest="command_home", help=argparse.SUPPRESS)
    web = sub.add_parser("web", help="serve the local approval page")
    web.add_argument("--port", type=int, default=8070)
    web.add_argument("--home", dest="command_home", help=argparse.SUPPRESS)
    ev = sub.add_parser("eval", help="run the 20-task planner evaluation")
    ev.add_argument("--config", choices=["fake", "ollama"], default="fake")
    ev.add_argument("--check", action="store_true", help="replay and verify recordings")
    ev.add_argument("--suite")
    ev.add_argument("--policy")
    ev.add_argument("--recording")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return EXIT_USAGE
    if args.cmd == "eval":
        from .eval import (
            DEFAULT_POLICY,
            DEFAULT_RECORDING,
            DEFAULT_SUITE,
            check_recording,
            render_markdown,
            run_eval,
            write_results,
        )

        suite = args.suite or DEFAULT_SUITE
        policy = args.policy or DEFAULT_POLICY
        recording = args.recording or DEFAULT_RECORDING
        try:
            if args.check:
                ok = check_recording(suite, policy, recording)
                print(
                    "recorded evaluation reproduced bench/results.md"
                    if ok
                    else "recorded evaluation differs from bench/results.md"
                )
                return 0 if ok else 1
            result = run_eval(args.config, suite, policy, recording)
            if args.config == "ollama":
                write_results(result)
            print(render_markdown(result), end="")
            return 0
        except (OSError, ValueError, RuntimeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    home = getattr(args, "command_home", None) or args.home
    if args.cmd == "web":
        from .web import serve

        if not 1 <= args.port <= 65535:
            print("error: --port must be between 1 and 65535", file=sys.stderr)
            return EXIT_USAGE
        serve(home, args.shop_db, args.port)
        return 0
    delay = float(os.environ.get("AGENT_EVENT_DELAY", "0") or 0)
    registry = default_registry()
    model = (
        OllamaModel(registry) if args.cmd == "run" and args.model == "ollama" else None
    )
    engine = Engine(
        home,
        model=model,
        registry=registry,
        shop_db=args.shop_db,
        delay=delay,
    )
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
            if args.model == "scripted" and not args.script:
                raise UsageError("--script is required with --model scripted")
            if args.model == "ollama" and args.script:
                raise UsageError("--script cannot be used with --model ollama")
            script = load_script(args.script) if args.script else []
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
        if args.cmd == "replay":
            from .replay import (
                recording_from_store,
                replay_run,
                write_regression_test,
            )

            result = replay_run(
                engine.store, args.run, shop_db=args.shop_db, registry=registry
            )
            print(result.message)
            if not result.identical:
                print(f"  original: {json.dumps(result.original, sort_keys=True)}")
                print(f"  replayed: {json.dumps(result.replayed, sort_keys=True)}")
                return 1
            if args.to_test:
                try:
                    fixture = write_regression_test(
                        args.to_test, recording_from_store(engine.store, args.run)
                    )
                except (OSError, ValueError) as exc:
                    raise UsageError(str(exc)) from None
                print(f"wrote {args.to_test} and {fixture}")
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
