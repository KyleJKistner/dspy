from __future__ import annotations

from types import SimpleNamespace

from dspy.cli import rlm as rlm_cli
from dspy.signatures.signature import ensure_signature


class _FakeLM:
    def __init__(self, *args, **kwargs):
        self.history = []
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return ["ok"]


class _FakeRLM:
    def __init__(self, signature, **kwargs):
        self.signature = ensure_signature(signature)
        self.kwargs = kwargs
        self._lm = kwargs.pop("_lm", None)

    def __call__(self, **inputs):
        outputs = {name: f"value:{name}" for name in self.signature.output_fields}
        outputs["answer"] = inputs.get("query", outputs.get("answer", ""))
        outputs["trajectory"] = [{}]
        return SimpleNamespace(**outputs)


def test_rlm_cli_prompt_shorthand(monkeypatch, capsys):
    fake_lm = _FakeLM()
    fake_lm.history.append({"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", lambda **kwargs: fake_lm)
    monkeypatch.setattr(rlm_cli.dspy, "RLM", _FakeRLM)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)

    rc = rlm_cli.main(["What is 2+2?"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "What is 2+2?" in out
    assert "total_tokens=15" in out


def test_rlm_cli_requires_json_for_multi_input(monkeypatch, capsys):
    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", _FakeLM)
    monkeypatch.setattr(rlm_cli.dspy, "RLM", _FakeRLM)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)

    rc = rlm_cli.main(["hello", "--signature", "a, b -> c"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "multi-input signatures" in err


def test_rlm_cli_json_mode(monkeypatch, capsys):
    fake_lm = _FakeLM()
    fake_lm.history.append({"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})

    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", lambda **kwargs: fake_lm)
    monkeypatch.setattr(rlm_cli.dspy, "RLM", _FakeRLM)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)

    rc = rlm_cli.main(["--json", "--inputs-json", '{"query":"ping"}'])
    out = capsys.readouterr().out

    assert rc == 0
    assert '"outputs"' in out
    assert '"usage"' in out


def test_build_chat_query():
    query = rlm_cli._build_chat_query([("hello", "world")], "next")
    assert "USER:\nhello" in query
    assert "ASSISTANT:\nworld" in query
    assert query.endswith("ASSISTANT:")


def test_rlm_cli_auto_tui(monkeypatch):
    fake_lm = _FakeLM()
    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", lambda **kwargs: fake_lm)
    monkeypatch.setattr(rlm_cli.dspy, "RLM", _FakeRLM)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(rlm_cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    captured = {}

    def fake_tui(args, rlm, lm, input_field, output_field):
        captured["mode"] = "rlm"
        captured["input_field"] = input_field
        captured["output_field"] = output_field
        return 0

    monkeypatch.setattr(rlm_cli, "_run_tui_session", fake_tui)

    rc = rlm_cli.main([])
    assert rc == 0
    assert captured["mode"] == "rlm"
    assert captured["input_field"] == "query"
    assert captured["output_field"] == "answer"


def test_rlm_cli_tui_mode_rlm(monkeypatch):
    fake_lm = _FakeLM()
    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", lambda **kwargs: fake_lm)
    monkeypatch.setattr(rlm_cli.dspy, "RLM", _FakeRLM)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(rlm_cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))

    captured = {}

    def fake_tui(args, rlm, lm, input_field, output_field):
        captured["mode"] = "rlm"
        captured["input_field"] = input_field
        captured["output_field"] = output_field
        return 0

    monkeypatch.setattr(rlm_cli, "_run_tui_session", fake_tui)

    rc = rlm_cli.main(["--mode", "rlm"])
    assert rc == 0
    assert captured["mode"] == "rlm"
    assert captured["input_field"] == "query"
    assert captured["output_field"] == "answer"


def test_render_transcript_lines_multiline():
    lines = rlm_cli._render_transcript_lines(
        [("you", "line1\nline2"), ("error", "boom")],
        width=50,
    )
    rendered = [line for line, _attr in lines]
    assert any("[YOU]" in line for line in rendered)
    assert any("line2" in line for line in rendered)
    assert any("[ERROR]" in line for line in rendered)


def test_startup_notices_explicit_file_access():
    args = SimpleNamespace(tui_theme="neo", tui_density="compact", cwd="/tmp/repo")

    rlm_notices = rlm_cli._startup_notices(args, "rlm")
    assert any("File access: ENABLED" in line for line in rlm_notices)
    assert any("/tmp/repo" in line for line in rlm_notices)

    chat_notices = rlm_cli._startup_notices(args, "chat")
    assert any("File access: DISABLED" in line for line in chat_notices)
    assert any("--mode rlm" in line for line in chat_notices)


def test_rlm_cli_workspace_tools_are_callable(monkeypatch, tmp_path):
    """Regression test: ensure workspace file tools don't break due to name shadowing."""

    fake_lm = _FakeLM()
    monkeypatch.setattr(rlm_cli.dspy, "CodexCLI", lambda **kwargs: fake_lm)

    captured = {}

    def fake_rlm_ctor(signature, **kwargs):
        captured["tools"] = kwargs.get("tools") or []
        return _FakeRLM(signature, **kwargs)

    monkeypatch.setattr(rlm_cli.dspy, "RLM", fake_rlm_ctor)
    monkeypatch.setattr(rlm_cli.dspy, "configure", lambda **kwargs: None)

    # Trigger RLM runtime construction (one-shot mode; the fake RLM short-circuits execution).
    rc = rlm_cli.main(["ping", "--cwd", str(tmp_path)])
    assert rc == 0

    tools = captured.get("tools") or []
    tool_by_name = {getattr(t, "__name__", ""): t for t in tools}

    assert "workspace_root" in tool_by_name
    assert "list_dir" in tool_by_name
    assert "read_text_file" in tool_by_name

    # Each tool should run without error under the provided --cwd.
    root = tool_by_name["workspace_root"]()
    assert str(tmp_path) == root
    assert tool_by_name["list_dir"](".") == []

    p = tmp_path / "hello.txt"
    p.write_text("hi", encoding="utf-8")
    assert tool_by_name["read_text_file"]("hello.txt") == "hi"


def test_rlm_run_stats_and_trace_preview():
    prediction = SimpleNamespace(
        trajectory=[
            {"code": "print('x')\nllm_query('a')", "output": "x"},
            {"code": "llm_query_batched(['b'])", "output": "done"},
        ]
    )

    stats = rlm_cli._build_rlm_run_stats(prediction, lm_calls_delta=7, max_depth=4)
    assert "steps=2" in stats
    assert "lm_calls=7" in stats
    assert "subquery_intents=2" in stats
    assert "depth_budget=3" in stats

    preview = rlm_cli._trajectory_preview_lines(prediction, max_steps=1)
    assert preview[0].startswith("Trajectory:")
    assert any("Step 2:" in line for line in preview)


def test_metrics_badge_from_stats():
    rlm_badge = rlm_cli._metrics_badge_from_stats(
        "Run stats: steps=4 lm_calls=9 subquery_intents=3 recursion=on depth_budget=2",
        "rlm",
    )
    assert "steps:4" in rlm_badge
    assert "calls:9" in rlm_badge
    assert "subq:3" in rlm_badge
    assert "depth:2" in rlm_badge

    chat_badge = rlm_cli._metrics_badge_from_stats(
        "Run stats: lm_calls=5 mode=chat (no trajectory)",
        "chat",
    )
    assert chat_badge == "calls:5"
