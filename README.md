# agent-orchestrator

Runs multi-step agents that split tasks, use permissioned tools and pause for human approval, with durable checkpoints, budgets and replayable runs (local models only).

## What it is

A small agent runtime where the interesting part is control, not the happy path. Every run is:

- **durable**: each step is an event in an append-only SQLite log; the checkpoint is derived from it, so a killed process resumes where it stopped without repeating finished tool calls;
- **governed**: tools declare a risk level (`safe`, `review`, `forbidden`); `review` calls pause the run until a human approves, edits or rejects them (or `policy.toml` auto-approves);
- **budgeted**: max steps, tool calls, model tokens and wall time per run;
- **local-first**: runs use a scripted fake or local Ollama `qwen2.5:3b`; the only model endpoint is
  `127.0.0.1:11434`. No LangGraph; see DECISIONS.

Tools only touch a per-run workspace (`<home>/runs/<id>/workspace`) and a read-only demo database; `send_email` appends to `outbox.jsonl` and never sends.

## Usage

```bash
uv run agent run --goal "Thank the top customer" --script examples/demo_script.json   # --home DIR or $AGENT_HOME, default ./.agent
uv run agent show <run>                                  # event timeline
uv run agent approve <run> --decision approve            # or reject | edit --args '{"to": "..."}'
uv run agent resume <run>                                # after a crash
uv run agent runs | cancel <run>
uv run agent replay <run>                                # fresh sandbox, recorded model output
uv run agent replay <run> --to-test tests/regressions/test_case.py
uv run agent web                                         # http://127.0.0.1:8070
```

Exit codes: 0 done, 1 failed, 2 usage error, 3 waiting for approval, 4 budget exceeded, 5 cancelled, 6 unknown run, 7 run busy (another process is driving it).
Budget flags on `run`: `--max-steps`, `--max-tool-calls`, `--max-tokens`, `--max-wall-seconds` (per invocation).
A script is a JSON list of actions: `{"action": "tool_call", "name": ..., "args": {...}}`, `{"action": "finish", "answer": ...}`, `{"action": "fail", "reason": ...}`.

Use the validated Ollama planner instead of a script:

```bash
uv run agent run --goal "Calculate 19 + 23" --model ollama
```

The planner receives the goal, compact bounded history, and only the run's tool schemas and risk
levels. It requests a JSON action schema with temperature 0 and seed 42. Common small-model shapes
(an implicit finish, numeric answer, or nested tool arguments) are normalized deterministically and
noted in the `model_call` event before validation. Every action and tool argument is validated before
use. Invalid output gets one repair turn containing the specific validator error; a failed repair is
one recoverable invalid action under the runtime's three-consecutive-errors rule. Ollama's prompt and
generation counts are charged to the token budget, including repair calls. Model configuration is
stored with the run, so `resume` and `approve` reconstruct the same planner after a process restart.

## Architecture

```text
CLI / localhost web page
          |
          v
 Engine state machine ----> risk policy / approval interrupt
    |              |
    |              +------> sandboxed tools ----> workspace, demo DB, fake outbox
    v
append-only SQLite events ----> pure reducer ----> derived checkpoint
    |
    +------> ReplayModel + recorded decisions ----> fresh sandbox ----> event comparison
```

The CLI and web page both call `Engine.decide`; schema validation, the per-run lock and execution
therefore have one implementation. Replay reconstructs the root and delegated child runs from their
events, feeds recorded model actions without contacting a model, replays recorded approval choices,
and re-executes tools against a fresh workspace seeded with the files that existed when the run
started.

Event comparison includes event type and deterministic event data, including tool name and args,
success flag, and a SHA-256 hash of each tool result. Sequence numbers, timestamps, run/call IDs,
workspace paths and elapsed-time-derived child wall budgets are excluded. The first mismatch is
reported. This detects changed tool behavior without pretending incidental process data is stable.

## Lead and worker delegation

A lead can return a `delegate` action with a narrow sub-goal, a subset of its tools and optional
child budget limits. A worker cannot delegate again, widen its tools or receive a forbidden tool.
The child's limits are capped by the parent's remaining steps, calls and tokens; actual child usage
is charged back to the parent. Parent and child have linked append-only logs and share the parent's
sandboxed workspace. A child approval pauses the parent, and `agent approve <parent>` resolves it.
Creation, approvals and completion are resumable across process crashes.

Example action:

```json
{"action":"delegate","task":"Calculate the subtotal","tools":["calc"],"budgets":{"max_steps":3,"max_tool_calls":2,"max_tokens":500}}
```

## Evaluation suite

`suite/tasks.yaml` contains 20 machine-checked tasks over the demo database, workspace, calculator
and outbox. It includes single-tool and multi-step work, delegation, approval, prompt injection in a
file and database row, forbidden actions, impossible goals and budget exhaustion. Evaluation tools
never make real network side effects. Approval answers are deterministic: the documented policy in
`suite/eval_policy.json` rejects by default and explicitly approves only the three safe cases that
exercise successful review tools. Injection cases are always rejected if they request email.

```bash
uv run agent eval --config fake       # deterministic harness/checker test
uv run agent eval --config ollama     # live local run; overwrites the recording/results
uv run agent eval --check             # fully offline replay of the committed recording
```

| Planner | Passed | Rate | Steps | Tool calls | Tokens | Repairs | Approval requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deterministic fake | 20/20 | 100% | — | — | — | — | — |
| `qwen2.5:3b` recorded sample | 9/20 | 45.0% (95% CI 25.8–65.8%) | 69 | 34 | 33,721 | 4 | 12 |

The corrected recorded `qwen2.5:3b` run had zero infrastructure exclusions. The deterministic fake
demonstrates the harness/checkers, not model quality. The original 3/20 M2 sample overstated model failure because its
finish and nested-call shapes were rejected and one checker mishandled a clean approval rejection;
those are harness/checker defects now covered by tests. The remaining 11 failures in the corrected
sample are recorded model behavior: wrong arithmetic/SQL, misuse of `fail`, repetition after success,
prompt-injection task reasoning errors, or poor delegation control. One caveat: `injection_db` is partly a harness gap, because the per-task `notes` table
(`notes(id, body)`) is not described in the `sql_query` tool, so the model had to guess its columns. Ollama infrastructure calls retry
twice with backoff; final infrastructure errors are recorded, labeled, and excluded from the success
denominator. Per-task evidence is in `bench/results.md`, with raw request/response/error records in
`bench/recorded/ollama.jsonl`.

`policy.toml` (in the home dir or `--policy`):

```toml
[[rule]]
tool = "send_email"
match = { to = "*@example.com" }   # anchored glob per arg; * never matches spaces, commas or newlines
decision = "approve"               # or "reject"
```

## Recorded M3 showcase

The committed `tests/regressions/test_ollama_demo.json` came from one local `qwen2.5:3b` run. It
identified the top customer, then made an unproductive delegation whose worker calculated `0` twice
and exhausted its small budget. The lead continued, wrote a usable draft, and paused twice. The file
write was approved; for the email, the model proposed the invalid placeholder
`customers/name@domain.com`, so the human used the web Edit-args path to replace only the recipient
with the database value `grace@example.com`. Selected real `agent show` output:

```text
$ agent show b06c6698
run b06c6698  status=done  steps=5  tool_calls=3  tokens=6599
#2    model_call         tool_call sql_query (991 tokens)
#4    tool_result        c1 sql_query ok=True: {"columns": ["id", "name", "total_revenue"], "rows": [[3, "Grace Hopper", 558.5]] ...
#5    model_call         delegate Compute the subtotal (1074 tokens)
#6    delegation_started worker 867469b3 task=Compute the subtotal tools=calc
#15   delegation_finished worker 867469b3 budget_exceeded
#17   tool_requested     c2 fs_write ... risk=review
#18   approval_requested c2 waiting for a human
#19   approval_decided   c2 approve by human
#22   tool_requested     c3 send_email ... risk=review
#23   approval_requested c3 waiting for a human
#24   approval_decided   c3 edit by human args={..., "to": "grace@example.com"}
#25   tool_result        c3 send_email ok=True: queued email to grace@example.com in outbox
#26   model_call         finish {"id": 3, "name": "Grace Hopper", "total_revenue": 558.5} (1388 tokens)
#27   finished           done: {"id": 3, "name": "Grace Hopper", "total_revenue": 558.5}

$ agent replay b06c6698 --to-test tests/regressions/test_ollama_demo.py
identical
wrote tests/regressions/test_ollama_demo.py and tests/regressions/test_ollama_demo.json
$ pytest -q tests/regressions/test_ollama_demo.py
1 passed
```

The query selected the customer's name and revenue but omitted the email column, and the model then
invented a recipient. The approval gate is what caught that unsafe proposal: no outbox write happened
until the human supplied the verified address. Replay preserves both the bad proposal and the human
edit, making the governed outcome reproducible without presenting the model's mistake as acceptable.

## Limits

- Replay proves deterministic engine/tool behavior for recorded outputs; it does not claim that a
  fresh model call would return the same text.
- Initial workspace files are embedded in the event recording. The read-only demo database is an
  external deterministic fixture and must still be available when replaying. Generated tests search
  their repository parents, `AGENT_SHOP_DB`, the current checkout and the source tree; installed use
  outside a checkout must pass `shop_db=...` or set `AGENT_SHOP_DB`.
- The web page is a single-user localhost review surface, not a remotely deployable auth system.
- SQLite plus `flock` targets one machine. Distributed workers would require a different store and
  lock implementation behind the same interfaces.
- The 3B local model is cheap and private but its 45% recorded suite result includes reasoning,
  tool-selection and delegation mistakes; approvals govern side effects, not answer quality.

## Setup

```bash
uv sync
```

## Development

Run the tests:

```bash
uv run pytest && uv run ruff check .
```

## How this repo is organised

- `AGENTS.md` – rules for coding agents (Claude Code reads it through `CLAUDE.md`)
- `DECISIONS.md` – why things are the way they are
- `CHANGELOG.md` – what changed in each release
- `suite/tasks.yaml` – 20 goals, deterministic fake scripts and machine checks
- `bench/results.md` – recorded local-model metrics, reproducible offline
