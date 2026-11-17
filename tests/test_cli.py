import json

import pytest

from agent_orchestrator import __version__
from agent_orchestrator.cli import main

from .conftest import SHOP_DB, finish, tc


@pytest.fixture
def run_cli(tmp_path, capsys):
    base = ["--home", str(tmp_path / "h"), "--shop-db", str(SHOP_DB)]

    def _run(*args):
        code = main([*base, *args])
        return code, capsys.readouterr()

    return _run


def write_script(tmp_path, actions):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(actions))
    return str(p)


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_no_command_is_usage_error():
    assert main([]) == 2


def test_run_approve_show_runs_exit_codes(run_cli, tmp_path):
    email = tc("send_email", to="a@b.c", subject="s", body="b")
    script = write_script(tmp_path, [email, finish("sent")])
    code, cap = run_cli("run", "--goal", "mail", "--script", script)
    assert code == 3 and "waiting_approval" in cap.out
    rid = cap.out.split()[1]
    code, cap = run_cli("runs")
    assert code == 0 and rid in cap.out
    code, cap = run_cli("approve", rid, "--decision", "edit", "--args", "{bad json")
    assert code == 2
    code, cap = run_cli("approve", rid, "--decision", "edit", "--args", '{"to": "x"}')
    assert code == 2 and "rejected" in cap.err
    code, cap = run_cli("approve", rid, "--decision", "approve")
    assert code == 0 and "sent" in cap.out
    code, cap = run_cli("show", rid)
    assert code == 0 and "approval_decided" in cap.out and "finished" in cap.out
    code, _ = run_cli("approve", rid, "--decision", "approve")
    assert code == 2


def test_exit_codes_failed_budget_cancel_unknown(run_cli, tmp_path):
    code, _ = run_cli(
        "run",
        "--goal",
        "g",
        "--script",
        write_script(tmp_path, [{"action": "fail", "reason": "no"}]),
    )
    assert code == 1
    s = write_script(tmp_path, [tc("calc", expression="1")] * 3)
    code, cap = run_cli("run", "--goal", "g", "--script", s, "--max-steps", "2")
    assert code == 4 and "max_steps" in cap.out
    email = write_script(
        tmp_path, [tc("send_email", to="a@b.c", subject="s", body="b")]
    )
    code, cap = run_cli("run", "--goal", "g", "--script", email)
    code, _ = run_cli("cancel", cap.out.split()[1])
    assert code == 5
    code, _ = run_cli("show", "nope")
    assert code == 6
    code, _ = run_cli("run", "--goal", "g", "--script", str(tmp_path / "missing.json"))
    assert code == 2
