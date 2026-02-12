import subprocess
from pathlib import Path

import pytest

import dspy


def _fake_completed_process(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_codex_cli_calls_exec_and_reads_last_message(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        out_file = cmd[cmd.index("--output-last-message") + 1]
        Path(out_file).write_text("final answer", encoding="utf-8")
        return _fake_completed_process(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = dspy.CodexCLI(working_dir=str(tmp_path))
    result = lm("What is 2+2?")

    assert result == ["final answer"]
    assert captured["cmd"][:2] == ["codex", "exec"]
    assert "--model" in captured["cmd"]
    assert "gpt-5.3-codex" in captured["cmd"]
    assert f'model_reasoning_effort="{lm.reasoning_effort}"' in captured["cmd"]
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert "ASSISTANT:" in captured["kwargs"]["input"]
    assert captured["kwargs"]["timeout"] == lm.timeout_seconds
    assert lm.history[-1]["usage"]["total_tokens"] > 0


def test_codex_cli_renders_chat_messages(monkeypatch):
    captured_prompt = {}

    def fake_run(cmd, **kwargs):
        captured_prompt["input"] = kwargs["input"]
        out_file = cmd[cmd.index("--output-last-message") + 1]
        Path(out_file).write_text("ok", encoding="utf-8")
        return _fake_completed_process(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = dspy.CodexCLI()
    result = lm(messages=[
        {"role": "system", "content": "Be terse."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        },
    ])

    assert result == ["ok"]
    rendered = captured_prompt["input"]
    assert "SYSTEM:\nBe terse." in rendered
    assert "USER:\nDescribe this image" in rendered
    assert "[image_url: https://example.com/cat.png]" in rendered


def test_codex_cli_supports_n_outputs(monkeypatch):
    outputs = iter(["first", "second"])
    call_count = {"value": 0}

    def fake_run(cmd, **kwargs):
        call_count["value"] += 1
        out_file = cmd[cmd.index("--output-last-message") + 1]
        Path(out_file).write_text(next(outputs), encoding="utf-8")
        return _fake_completed_process(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = dspy.CodexCLI()
    result = lm("Generate twice", n=2)

    assert result == ["first", "second"]
    assert call_count["value"] == 2


def test_codex_cli_raises_on_command_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _fake_completed_process(cmd, returncode=1, stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = dspy.CodexCLI()
    with pytest.raises(RuntimeError, match="codex exec failed"):
        lm("fail please")

