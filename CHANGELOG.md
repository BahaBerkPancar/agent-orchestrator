# Changelog

## Unreleased
- Project created.
- M1 core runtime: durable runs (SQLite append-only event log + derived checkpoint), governed tools with risk levels
  (`fs_read`, `fs_write`, `fs_list`, `calc`, `sql_query`, `send_email` to an outbox file), approval interrupts
  (approve / edit / reject, `policy.toml` auto-approve), budgets, kill-and-resume, scripted fake model.
- `agent` CLI: `run`, `resume`, `approve`, `cancel`, `show`, `runs`; `examples/demo.sh`.
