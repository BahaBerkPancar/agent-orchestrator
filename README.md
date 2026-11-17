# agent-orchestrator

Runs multi-step agents that split tasks, use permissioned tools and pause for human approval, with durable checkpoints, budgets and replayable runs (local models only).

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
