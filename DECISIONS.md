# Decisions

Newest first. One entry per decision: date, what was decided, why, and what was rejected.

## 2026-11-17 – M1 runtime: own state machine, not LangGraph
Spike: `uv pip install --offline --dry-run langgraph langgraph-checkpoint-sqlite jsonschema` fails ("langgraph was not
found in the cache", same for jsonschema); the build is offline, so LangGraph could not even be tried. Decision: a small
own state machine (~250 lines in `engine.py` + `state.py`). Against the criteria: durable resume (SQLite, one transaction per
event), approval interrupts with approve/edit/reject (a `waiting_approval` status plus `agent approve`), deterministic replay
(the event log is the source of truth; the checkpoint is `reduce(events)`), and zero runtime dependencies. LangGraph would add
a large dependency tree and its checkpointer stores graph snapshots, not an append-only event log we control, so replay in M3
would need our own log anyway. Rejected: LangGraph (not installable offline, replay mismatch). Revisit only if M2's
lead/worker delegation needs graph features we end up rebuilding.

## 2026-11-17 – Event log is truth, checkpoint is derived
`events` (SQLite, AUTOINCREMENT seq, triggers forbid UPDATE/DELETE) is the only thing written by the engine; each append also
stores `reduce(state, event)` in `checkpoints` in the same transaction. `Store.rebuild` replays events and a test asserts it
equals the stored checkpoint. A model output is logged first (`model_call`, kept in `state.pending_action`) and acted on after,
so a crash between the two never skips or repeats a scripted action: the scripted model's position is `state.step`.
A `started` event was added to the required event types so the run's goal, budgets and script live in the log too.

## 2026-11-17 – Idempotency and crash windows
Each tool call gets key `<run_id>:<call_id>`. A tool_result is logged after the tool runs, so a kill in between re-runs the
tool on resume; `send_email` checks the outbox for its key and does not append twice (tested explicitly). `fs_write` is
idempotent by nature. Tools without that property must check the key before adding new tools.

## 2026-11-17 – Smaller choices
- JSON schema: a ~40-line own validator (type/required/properties/enum/min/max/items/additionalProperties) because
  `jsonschema` and pydantic were not available offline. Swap in `jsonschema` later if schemas get richer.
- Wall-time budget is measured per process invocation (`run`/`resume`/`approve`), so time spent waiting for a human or
  downtime after a crash does not eat the budget. Steps, tool calls and tokens are cumulative over the whole run.
- Three invalid model actions in a row (malformed, unknown tool, bad args, delegate stub) fail the run; the errors are fed back
  as observations first so a real model can repair.
- `sql_query` is read-only three ways: SELECT-only prefix and single statement, `mode=ro` connection plus `query_only`,
  a SQLite authorizer that denies everything except SELECT/READ/FUNCTION, plus limits: 2 s deadline (progress handler), 1000 rows, ~4 KB of JSON output, blobs rendered as `<blob N bytes>`.
- `calc` caps every intermediate result (4096 bits, finite floats) and pre-checks `**` so nested powers never get computed.
- Concurrency: an exclusive non-blocking `flock` per run (`<home>/locks/<run>.lock`) is held for the whole run/resume/approve/cancel; a second process exits 7 "run is busy". The outbox append is additionally atomic (flock + O_APPEND single write + dedup under the lock). The policy file path is stored in the run's `started` event and reused by resume/approve.
- Policy globs are anchored and `*`/`?` never match whitespace, commas, semicolons or angle brackets; `send_email.to` is a schema-validated single address.
- Postgres/Redis are not used: SQLite is enough for a single-machine demo; `Store` is the only module that knows SQL.

## 2026-11-17 – Project created from the `python-app` template
Why: same layout, test setup and CI as the other projects, so nothing has to be set up by hand.
