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

The corrected recorded `qwen2.5:3b` run scored **9/20 (45.0%)**, Wilson 95% CI **25.8–65.8%**. It
used 69 steps, 34 tool calls, 33,721 model tokens, four repair turns and 12 approval requests, with
zero infrastructure exclusions. The deterministic fake remains **20/20** and demonstrates the
harness/checkers, not model quality. The original 3/20 M2 sample overstated model failure because its
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

## Demo run (`examples/demo.sh`)

```
$ agent run --goal "Thank the top customer" --script examples/demo_script.json
run f21f4e96 status=waiting_approval
  waiting: send_email {"to": "grace@example.com", "subject": "Thank you", "body": "Hi Grace, thanks for being our top customer: 558.5 in October."}
  approve with: agent approve f21f4e96 --decision approve|reject|edit [--args '{...}']

...
$ agent show f21f4e96
run f21f4e96  status=done  steps=6  tool_calls=5  tokens=116
goal: Thank the top customer
#1    06:02:28  started            goal: Thank the top customer
#2    06:02:28  model_call         tool_call sql_query (40 tokens)
#3    06:02:28  tool_requested     c1 sql_query {"query": "SELECT c.name, c.email, SUM(o.qty * p.price) AS revenue FROM orders o JOIN customers c ON c
#4    06:02:28  tool_result        c1 sql_query ok=True: {"columns": ["name", "email", "revenue"], "rows": [["Grace Hopper", "grace@example.com", 558.
#5    06:02:28  model_call         tool_call calc (15 tokens)
#6    06:02:28  tool_requested     c2 calc {"expression": "2 * 189 + 32 + 3 * 49.5"} risk=safe
#7    06:02:28  tool_result        c2 calc ok=True: 558.5
#8    06:02:28  model_call         tool_call fs_write (19 tokens)
#9    06:02:28  tool_requested     c3 fs_write {"content": "Hi Grace, thanks for being our top customer: 558.5 in October.", "path": "drafts/thanks.md
#10   06:02:28  tool_result        c3 fs_write ok=True: wrote 62 chars to drafts/thanks.md
#11   06:02:28  model_call         tool_call fs_read (7 tokens)
#12   06:02:28  tool_requested     c4 fs_read {"path": "drafts/thanks.md"} risk=safe
#13   06:02:28  tool_result        c4 fs_read ok=True: Hi Grace, thanks for being our top customer: 558.5 in October.
#14   06:02:28  model_call         tool_call send_email (22 tokens)
#15   06:02:28  tool_requested     c5 send_email {"body": "Hi Grace, thanks for being our top customer: 558.5 in October.", "subject": "Thank you", "t
#16   06:02:28  approval_requested c5 waiting for a human
#17   06:02:28  approval_decided   c5 approve by human
#18   06:02:28  tool_result        c5 send_email ok=True: queued email to grace@example.com in outbox
#19   06:02:28  model_call         finish Drafted and sent a thank-you to Grace Hopper (558.5 revenue). (13 tokens)
#20   06:02:28  finished           done: Drafted and sent a thank-you to Grace Hopper (558.5 revenue).

$ cat workspace/outbox.jsonl
{"idempotency_key": "f21f4e96:c5", "to": "grace@example.com", "subject": "Thank you", "body": "Hi Grace, thanks for being our top customer: 558.5 in O```

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
