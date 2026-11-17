import argparse

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agent-orchestrator",
        description="Runs multi-step agents that split tasks, use permissioned tools and pause for human approval, with durable checkpoints, budgets and replayable runs (local models only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    print("Hello from agent-orchestrator")
    return 0
