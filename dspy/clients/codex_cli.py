from __future__ import annotations

import asyncio
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from litellm.utils import Choices, Message, ModelResponse

from dspy.clients.base_lm import BaseLM


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
        cache: bool = False,
        extra_config: dict[str, Any] | None = None,
    ):
        super().__init__(model=model, model_type="chat", temperature=1.0, max_tokens=None, cache=cache)
        self.command = command
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.working_dir = working_dir
        self.extra_config = dict(extra_config or {})

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
        preamble = (
            "You are being used as a pure language model backend for DSPy.\n"
            "Respond to the conversation directly.\n"
            "Do not run shell commands or tools.\n"
            "Return only the assistant response text."
        )
        return f"{preamble}\n\n{conversation}"

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
            "--output-last-message",
            output_file,
            "--skip-git-repo-check",
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
        ]
        for key, value in self.extra_config.items():
            command.extend(["-c", f"{key}={self._to_toml_literal(value)}"])
        command.append("-")
        return command

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    def _run_once(self, prompt_text: str) -> tuple[str, dict[str, int]]:
        with tempfile.NamedTemporaryFile(prefix="dspy_codex_", suffix=".txt", delete=False) as temp_output:
            output_path = temp_output.name

        cmd = self._build_command(output_path)
        completed = subprocess.run(
            cmd,
            input=prompt_text,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=self.working_dir,
            check=False,
        )

        output_text = Path(output_path).read_text(encoding="utf-8").strip()
        Path(output_path).unlink(missing_ok=True)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"codex exec failed with exit code {completed.returncode}: {stderr}")

        if not output_text:
            output_text = (completed.stdout or "").strip()

        usage = {
            "prompt_tokens": self._estimate_tokens(prompt_text),
            "completion_tokens": self._estimate_tokens(output_text),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return output_text, usage

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

