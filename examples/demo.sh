#!/usr/bin/env bash
# Scripted demo: multi-step task, pauses for an email approval, approved from the CLI.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export AGENT_HOME="$(mktemp -d)/agent"
agent() { (cd "$ROOT" && uv run --offline --quiet agent "$@"); }

echo '$ agent run --goal "Thank the top customer" --script examples/demo_script.json'
OUT=$(agent run --goal "Thank the top customer" --script "$ROOT/examples/demo_script.json"); echo "$OUT"
RUN=$(echo "$OUT" | head -1 | awk '{print $2}')
echo; echo "\$ agent show $RUN"; agent show "$RUN"
echo; echo "\$ agent approve $RUN --decision approve"; agent approve "$RUN" --decision approve
echo; echo "\$ agent show $RUN"; agent show "$RUN"
echo; echo "\$ cat workspace/outbox.jsonl"; cat "$AGENT_HOME/runs/$RUN/workspace/outbox.jsonl"
