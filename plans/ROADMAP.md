# Build plan: agent-orchestrator

Written 2026-11-17 (away run). Source: vault `knowledge/project-ideas-backlog.md`, "Saved for later":
"Agent Orchestration System: agents that divide tasks, use tools, remember context and ask for human
approval. Stack: Python, LangGraph, PostgreSQL, Redis. Crowded; mostly framework wiring."

## How it stands out (not just framework wiring)
The interesting part of agents in production is **control**, not the happy path. This project is a
small orchestrator where every run is:
- **durable**: checkpointed after every step; a crash or restart resumes exactly where it stopped;
- **governed**: tools declare a risk level; risky calls pause for human approval (approve / edit /
  reject) through a CLI and a tiny web page; per-run budgets (steps, tool calls, tokens, wall time);
- **replayable**: a run's event log can be replayed deterministically with recorded model outputs,
  so a failure can be reproduced and turned into a test (ties into failure-forensics);
- **measured**: a task suite with success criteria and an eval report (success rate, steps, approvals,
  budget hits) per planner configuration.
Models are local (Ollama `qwen2.5:3b`) or a scripted fake; no paid API calls.

## Ground rules
- No paid APIs; Ollama on 127.0.0.1:11434 only; tests use a scripted fake model.
- Tools are sandboxed: a temp workspace dir, an allow-listed toolset (read/write files in the
  workspace, a calculator, a SQLite query tool on a demo DB, a fake "send email" that only writes to
  an outbox file — never real network side effects).
- Never read `.env`. Storage: SQLite (checkpoints + event log); Postgres/Redis are not needed for a
  single-machine demo — DECISIONS explains, with the interface kept swappable.
- LangGraph: use it only if it earns its place (graph + checkpointer + interrupts); otherwise a
  small own state machine. Decide in M1 and justify.
- Builders: Codex (Sonnet fallback); reviews by the `reviewer` agent; hub commits locally; created
  `--local`, no GitHub repo until MF says.

## Milestones
| # | Name | Contents |
|---|------|----------|
| M1 | Core runtime | Run state machine (plan → act → observe → reflect → done/fail), tool registry with JSON-schema args + risk levels, approval interrupts, budgets, SQLite checkpoints + append-only event log, resume after crash, scripted fake model, `agent run/resume/approve/show` CLI. |
| M2 | Real planner + task suite | Ollama planner (qwen2.5:3b) with structured tool calls (JSON mode + validation + repair), sub-task decomposition (a lead agent delegating to worker agents with narrower toolsets), 20-task suite with checkers, eval report. |
| M3 | Replay + approval UI + showcase | Deterministic replay from the event log, `agent replay --to-test`, a minimal local approval page, README with a recorded demo run, limits. |
