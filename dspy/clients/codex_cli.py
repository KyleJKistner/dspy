from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

from litellm.utils import Choices, Message, ModelResponse

from dspy.clients.base_lm import BaseLM


class CodexCLICancelled(RuntimeError):
    """Raised when a CodexCLI request is cancelled via `cancel_event`."""


class CodexCLI(BaseLM):
    """DSPy LM client backed by the local `codex exec` CLI.

    This client is useful when you want to use a ChatGPT subscription-backed
    Codex CLI session instead of API keys. It shells out to `codex exec` for
    each LM request and converts the response into DSPy's LM response format.
    """

    def __init__(
        self,
        model: str = "gpt-5.3-codex",
        reasoning_effort: str = "xhigh",
        command: str = "codex",
        timeout_seconds: float = 300.0,
        working_dir: str | None = None,
        prompt_mode: Literal["dspy", "chat"] = "dspy",
        sandbox: Literal["read-only", "workspace-write", "danger-full-access"] | None = None,
        ask_for_approval: Literal["untrusted", "on-failure", "on-request", "never"] | None = None,
        codex_cd: str | None = None,
        cache: bool = False,
        extra_config: dict[str, Any] | None = None,
    ):
        super().__init__(model=model, model_type="chat", temperature=1.0, max_tokens=None, cache=cache)
        self.command = command
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.working_dir = working_dir
        if prompt_mode not in {"dspy", "chat"}:
            raise ValueError(f"prompt_mode must be 'dspy' or 'chat', got {prompt_mode!r}")
        self.prompt_mode = prompt_mode
        self.sandbox = sandbox
        self.ask_for_approval = ask_for_approval
        self.codex_cd = codex_cd
        self.extra_config = dict(extra_config or {})
        # Optional cancellation token. If set, `forward()` will try to terminate the underlying
        # `codex exec` subprocess promptly and raise CodexCLICancelled.
        self.cancel_event: threading.Event | None = None

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text"}:
                        parts.append(str(item.get("text", "")))
                    elif item_type in {"image_url", "input_image"}:
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")
                        parts.append(f"[image_url: {image_url}]")
                    elif item_type in {"input_audio", "audio"}:
                        input_audio = item.get("input_audio", {})
                        if not isinstance(input_audio, dict):
                            input_audio = {}
                        parts.append(f"[audio: format={input_audio.get('format', 'unknown')}]")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _render_prompt(self, prompt: str | None, messages: list[dict[str, Any]] | None) -> str:
        normalized_messages = messages or [{"role": "user", "content": prompt or ""}]
        lines: list[str] = []
        for msg in normalized_messages:
            role = str(msg.get("role", "user")).upper()
            text = self._content_to_text(msg.get("content", ""))
            lines.append(f"{role}:\n{text}".strip())

        lines.append("ASSISTANT:")
        conversation = "\n\n".join(lines).strip()
        if self.prompt_mode == "chat":
            preamble = (
                "You are being used as a pure language model backend for DSPy.\n"
                "Respond to the conversation directly.\n"
                "Do not run shell commands or tools.\n"
                "Return only the assistant response text."
            )
            return f"{preamble}\n\n{conversation}"

        return conversation

    @staticmethod
    def _to_toml_literal(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            return json.dumps(value)
        return json.dumps(str(value))

    def _build_command(self, output_file: str) -> list[str]:
        command = [
            self.command,
            "exec",
            "--model",
            self.model,
        ]
        if self.sandbox is not None:
            command.extend(["--sandbox", self.sandbox])
        if self.ask_for_approval is not None:
            command.extend(["--ask-for-approval", self.ask_for_approval])
        if self.codex_cd is not None:
            command.extend(["-C", self.codex_cd])
        # Request JSONL events on stdout so we can parse accurate usage (when available).
        command.append("--json")
        command.extend([
            "--output-last-message",
            output_file,
            "--skip-git-repo-check",
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
        ])
        for key, value in self.extra_config.items():
            command.extend(["-c", f"{key}={self._to_toml_literal(value)}"])
        command.append("-")
        return command

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @staticmethod
    def _parse_jsonl_events(stdout_text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in (stdout_text or "").splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events

    @staticmethod
    def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, int] | None:
        for event in reversed(events):
            if event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            try:
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
            except (TypeError, ValueError):
                continue
            if input_tokens or output_tokens:
                return {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
        return None

    @staticmethod
    def _text_from_events(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            if event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if item.get("type") != "agent_message":
                continue
            text = item.get("text")
            if text is None:
                continue
            return str(text)
        return ""

    def _terminate_proc(self, proc: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                # We start a new session, so the process group is pid.
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            try:
                proc.terminate()
            except Exception:
                return

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                return

    def _run_once_cancellable(self, cmd: list[str], prompt_text: str) -> tuple[int, str, str]:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.working_dir,
            start_new_session=(os.name == "posix"),
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _reader(stream: Any, sink: list[str]) -> None:
            try:
                for line in stream:
                    sink.append(line)
            except Exception:
                return

        t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.stdin.write(prompt_text)
            proc.stdin.close()
        except Exception:
            # If stdin write fails, fall through to wait and report.
            try:
                proc.stdin.close()
            except Exception:
                pass

        start = time.monotonic()
        cancelled = False
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                cancelled = True
                self._terminate_proc(proc)
                break

            if self.timeout_seconds and self.timeout_seconds > 0:
                if (time.monotonic() - start) > self.timeout_seconds:
                    self._terminate_proc(proc)
                    break

            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.05)

        if cancelled:
            # Give a small grace window, then kill.
            try:
                proc.wait(timeout=0.5)
            except Exception:
                self._kill_proc(proc)
            raise CodexCLICancelled("codex exec cancelled")

        # If we terminated due to timeout, ensure the process is gone.
        if self.timeout_seconds and self.timeout_seconds > 0 and (time.monotonic() - start) > self.timeout_seconds:
            try:
                proc.wait(timeout=0.5)
            except Exception:
                self._kill_proc(proc)
            raise TimeoutError(f"codex exec timed out after {self.timeout_seconds}s")

        rc = proc.wait()
        t_out.join(timeout=0.5)
        t_err.join(timeout=0.5)
        return rc, "".join(stdout_chunks), "".join(stderr_chunks)

    def _run_once(self, prompt_text: str) -> tuple[str, dict[str, int]]:
        with tempfile.NamedTemporaryFile(prefix="dspy_codex_", suffix=".txt", delete=False) as temp_output:
            output_path = temp_output.name

        cmd = self._build_command(output_path)
        try:
            if self.cancel_event is not None:
                returncode, stdout, stderr = self._run_once_cancellable(cmd, prompt_text)
            else:
                completed = subprocess.run(
                    cmd,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    cwd=self.working_dir,
                    check=False,
                )
                returncode, stdout, stderr = completed.returncode, completed.stdout or "", completed.stderr or ""

            output_text = ""
            try:
                output_text = Path(output_path).read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                output_text = ""

            events = self._parse_jsonl_events(stdout)
            if not output_text:
                output_text = self._text_from_events(events).strip()
            if not output_text:
                output_text = (stdout or "").strip()

            if returncode != 0:
                raise RuntimeError(f"codex exec failed with exit code {returncode}: {(stderr or '').strip()}")

            usage = self._usage_from_events(events)
            if usage is None:
                usage = {
                    "prompt_tokens": self._estimate_tokens(prompt_text),
                    "completion_tokens": self._estimate_tokens(output_text),
                }
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            return output_text, usage
        finally:
            Path(output_path).unlink(missing_ok=True)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        prompt_text = self._render_prompt(prompt, messages)
        num_outputs = int(kwargs.get("n", 1) or 1)

        all_outputs: list[str] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for _ in range(num_outputs):
            output_text, usage = self._run_once(prompt_text)
            all_outputs.append(output_text)
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            total_usage["total_tokens"] += usage["total_tokens"]

        choices = [Choices(message=Message(role="assistant", content=output)) for output in all_outputs]
        return ModelResponse(choices=choices, usage=total_usage, model=self.model)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        return await asyncio.to_thread(self.forward, prompt=prompt, messages=messages, **kwargs)
