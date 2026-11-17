# Changelog

## Unreleased
- Project created.
- M1 core runtime: durable runs (SQLite append-only event log + derived checkpoint), governed tools with risk levels
  (`fs_read`, `fs_write`, `fs_list`, `calc`, `sql_query`, `send_email` to an outbox file), approval interrupts
  (approve / edit / reject, `policy.toml` auto-approve), budgets, kill-and-resume, scripted fake model.
- `agent` CLI: `run`, `resume`, `approve`, `cancel`, `show`, `runs`; `examples/demo.sh`.
- M2 local Ollama planner (`qwen2.5:3b`) with bounded history, strict JSON/action and tool-schema
  validation, one validator-guided repair, and model token/repair budget accounting.
- Durable lead-to-worker delegation with narrower toolsets, capped child budgets, parent-linked
  events, approval propagation and parent/child crash recovery.
- A 20-task machine-checked suite, fixed eval approval policy, Wilson confidence intervals, and
  response/error recordings reproducible offline with `agent eval --check`.
- M2 review fixes: action-schema output, deterministic implicit-finish/nested-call normalization,
  recoverable failed repairs, persisted planner configuration across CLI resume/approval, bounded
  Ollama infrastructure retries/exclusions, reject-by-default eval approvals, fair refusal checkers,
  remaining-wall-time child budgets, and mid-child crash recovery coverage.
- M3 deterministic run-tree replay with recorded model outputs and approvals, fresh seeded
  workspaces, tool re-execution, first-divergence reporting, and child-run support.
- `agent replay --to-test` generates a readable pytest plus adjacent JSON recording; the committed
  local-Ollama showcase regression replays offline.
- `agent web` serves a localhost-only light/dark run timeline and approval UI with escaping, CSP,
  exact loopback Host validation, bounded per-session CSRF state, edited-argument schema validation
  and the engine's run lock.
- README showcase adds the architecture, replay guarantees, M2 evaluation table, recorded M3
  transcript and explicit limitations.
