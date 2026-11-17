# agent-orchestrator

Runs multi-step agents that split tasks, use permissioned tools and pause for human approval, with durable checkpoints, budgets and replayable runs (local models only).

## What it is

A small agent runtime where the interesting part is control, not the happy path. Every run is:

- **durable**: each step is an event in an append-only SQLite log; the checkpoint is derived from it, so a killed process resumes where it stopped without repeating finished tool calls;
- **governed**: tools declare a risk level (`safe`, `review`, `forbidden`); `review` calls pause the run until a human approves, edits or rejects them (or `policy.toml` auto-approves);
- **budgeted**: max steps, tool calls, model tokens and wall time per run;
- **offline**: M1 is driven by a scripted fake model (no network, no LLM). Ollama arrives in M2. No LangGraph; see DECISIONS.

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
