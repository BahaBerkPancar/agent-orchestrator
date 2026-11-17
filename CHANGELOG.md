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
