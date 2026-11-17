# Decisions

Newest first. One entry per decision: date, what was decided, why, and what was rejected.

## 2026-11-17 – Replay compares deterministic event semantics in a fresh sandbox
Replay records every model-call payload and approval decision, restores the files present at run
start, reuses recorded child IDs internally, and re-executes tools in a temporary home. Comparison
keeps event type and deterministic data (notably tool name/args/ok plus a SHA-256 output hash) while
excluding timestamps, SQLite sequence numbers, IDs, workspace paths and elapsed-time-derived child
wall budgets. Two crash-boundary representations are also normalized: `delegation_resumed` is
process bookkeeping, and a successful `send_email` result is equivalent whether it says the outbox
write was queued or already present after idempotent crash recovery. The requested/edited arguments
are still compared, so this does not hide recipient changes. Generated regressions keep the recording
in a readable adjacent JSON fixture rather than a Python blob. Rejected: comparing raw event rows (false divergence on incidental process data),
mocking tool results (would not detect regressions), and calling the model again (not deterministic).

## 2026-11-17 – Approval UI uses the standard library and the engine's decision path
The page is served only on `127.0.0.1`, with dependency-free `http.server`, escaped content, no CDN,
a restrictive CSP and a random per-session CSRF token. Before minting a session or reading a request
body, it requires an exact `127.0.0.1:<port>` or `localhost:<port>` Host header to block DNS
rebinding; the in-memory session LRU is capped at 1,000. Approve, reject and edited-args POSTs call
`Engine.decide`, so schema validation, forbidden-risk checks and the run lock remain single-source.
Rejected: FastAPI for this local two-page surface (an unnecessary runtime dependency) and a separate
web mutation layer (would drift from CLI governance).

## 2026-11-17 – M2 planner uses an Ollama action schema plus one validator-guided repair
The local planner calls `/api/generate` at `127.0.0.1:11434` with a JSON schema in `format`,
temperature 0 and seed 42. Its prompt contains the goal, a 6,000-character bounded history, worked
examples, and the allowed tools' schemas and risks. Deterministic normalization accepts implicit
finishes, stringifies answer values and unwraps two observed nested tool-call shapes; every change is
noted in the model event. The runtime then parses the existing M1 action types and validates tool
arguments with the execution registry. One invalid response gets exactly one repair call with the
specific validator error; a second invalid response is one normal invalid-action event, not an
immediate terminal failure. Both calls' token and repair counts are budgeted. Rejected: Ollama's
native tool-calling API, because one provider-neutral action envelope also covers delegation, finish
and fail; unbounded repair, because a weak model could consume the run budget without progress.

## 2026-11-17 – Delegation is a linked child run with capped, charged-back usage
A delegate event is durable before child creation, so resume can recover the parent-to-child crash
window without orphaning or skipping work. The worker shares the parent's workspace but has a
separate event log, role, tool subset and budget. Requested limits are capped at the parent's
remaining steps, calls and tokens; completed child usage is charged back before the lead continues.
Workers cannot delegate, tools cannot be widened, and inherently forbidden tools cannot cross the
boundary. Child approvals set the parent to `waiting_approval` and are decided through the parent.
Rejected: an in-memory worker call (not crash-safe) and independent child budgets (could exceed the
parent's governance envelope).

## 2026-11-17 – Evaluation recordings include responses, retries and infra classification
The 20-task suite uses machine checkers and a fixed approval policy. A live Ollama run writes each
request/result to JSONL; each transport failure is recorded and retried twice with backoff. A final
infra error is labeled and excluded from the model-success denominator. `agent eval --check` verifies
both `bench/results.json` and its Markdown rendering offline. After fixing shape handling and unfair
checkers, the one replacement qwen2.5:3b sample scored 9/20 (45.0%, Wilson 95% CI 25.8–65.8%) with no
infra exclusions. Rejected: reporting the fake model's 20/20 as model quality, counting infra failure
as model failure, and rerunning until a flattering sample appeared. The fake configuration proves
deterministic engine/checker behavior only.

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
