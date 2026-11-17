from pathlib import Path

from agent_orchestrator.replay import assert_recorded_fixture


def test_ollama_demo(tmp_path):
    fixture = Path(__file__).with_name('test_ollama_demo.json')
    assert_recorded_fixture(fixture, tmp_path)
