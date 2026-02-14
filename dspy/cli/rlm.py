from __future__ import annotations

import argparse
import curses
import json
import locale
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import textwrap
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import dspy
from dspy.primitives.local_interpreter import LocalInterpreter
from dspy.signatures.signature import ensure_signature


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="Run DSPy RLM with CodexCLI + sandboxed interpreter defaults.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text for one-shot mode.")
    parser.add_argument("--tui", action="store_true", help="Launch interactive TUI mode.")
    parser.add_argument("--no-history", action="store_true", help="In TUI mode, do not include prior turns in each query.")
    parser.add_argument(
        "--mode",
        choices=["auto", "rlm", "chat"],
        default="rlm",
        help=(
            "Execution mode. 'rlm' runs the code loop; by default it uses a sandboxed interpreter and "
            "workspace file tools. 'chat' is text-only."
        ),
    )
    parser.add_argument(
        "--tui-theme",
        choices=["codex", "neo", "amber", "mono", "cyber"],
        default="codex",
        help="TUI color theme for interactive mode.",
    )
    parser.add_argument(
        "--tui-density",
        choices=["compact", "cozy"],
        default="compact",
        help="Transcript spacing density for interactive mode.",
    )
    parser.add_argument(
        "--tui-style",
        choices=["claude", "codex", "classic"],
        default="claude",
        help="TUI layout style for interactive mode.",
    )
    parser.add_argument(
        "--tui-charset",
        choices=["auto", "ascii", "unicode"],
        default="auto",
        help="TUI border charset. 'unicode' enables rounded box-drawing borders; 'ascii' forces +-|.",
    )
    parser.add_argument(
        "--no-boot-animation",
        action="store_true",
        help="Disable the short ASCII boot animation when launching the TUI.",
    )

    parser.add_argument("--signature", default="query -> answer", help="DSPy signature string.")
    parser.add_argument("--inputs-json", help="JSON object with all signature inputs.")
    parser.add_argument("--inputs-file", help="Path to JSON file with all signature inputs.")
    parser.add_argument("--output-field", help="Which output field to print in one-shot text mode.")
    parser.add_argument("--json", action="store_true", help="Print JSON output instead of plain text (one-shot mode).")
    parser.add_argument("--quiet", action="store_true", help="Suppress usage summary in one-shot text mode.")

    parser.add_argument("--model", default="gpt-5.3-codex", help="Codex model name.")
    parser.add_argument("--reasoning-effort", default="xhigh", help="Codex reasoning effort.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="Codex exec timeout.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for codex exec.")
    parser.add_argument("--codex-command", default="codex", help="Codex executable name/path.")
    parser.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="read-only",
        help="Sandbox policy for any model-generated shell commands inside codex exec.",
    )
    parser.add_argument(
        "--codex-prompt-mode",
        choices=["dspy", "chat"],
        default="chat",
        help="Prompt mode for codex exec. 'chat' adds a 'do not run shell commands/tools' preamble.",
    )
    parser.add_argument(
        "--codex-enable-shell-tool",
        action="store_true",
        help=(
            "Allow Codex to use its shell tool (model-generated shell commands). "
            "Disabled by default to prevent keychain/network/process access via shell."
        ),
    )
    parser.add_argument(
        "--codex-approval",
        choices=["untrusted", "on-failure", "on-request", "never"],
        default=None,
        help=(
            "Codex command approval policy. Default is 'never' when shell tool is disabled, "
            "and 'untrusted' when enabled."
        ),
    )
    parser.add_argument(
        "--codex-trust-level",
        choices=["untrusted", "trusted"],
        default="untrusted",
        help="Temporary trust level override for this workspace when invoking codex exec.",
    )

    parser.add_argument("--max-depth", type=int, default=2, help="RLM max recursion depth.")
    parser.add_argument("--max-iterations", type=int, default=6, help="RLM max iterations.")
    parser.add_argument("--max-llm-calls", type=int, default=120, help="RLM max llm_query* calls.")
    parser.add_argument("--max-time", type=float, default=900.0, help="RLM max wall-clock time (seconds).")
    parser.add_argument("--max-tokens", type=int, default=5_000_000, help="RLM max token budget (estimated).")
    parser.add_argument("--max-cost", type=float, default=None, help="Optional RLM max cost budget.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose RLM logging.")
    parser.add_argument(
        "--unsafe-local-interpreter",
        action="store_true",
        help=(
            "Use UNSANDBOXED host Python execution for the RLM REPL (LocalInterpreter). "
            "This allows arbitrary file/network/process access from model-generated code."
        ),
    )
    parser.add_argument(
        "--unsafe-local-subcalls",
        action="store_true",
        help=(
            "Use UNSANDBOXED host Python execution (LocalInterpreter) for recursive llm_query() "
            "subcalls when max_depth > 1. Default keeps subcalls sandboxed."
        ),
    )
    return parser


def _should_launch_tui(args: argparse.Namespace) -> bool:
    if args.tui:
        return True

    if args.prompt is not None or args.inputs_json or args.inputs_file or args.json:
        return False

    return sys.stdin.isatty()


def _resolve_mode(args: argparse.Namespace, launch_tui: bool) -> Literal["rlm", "chat"]:
    if args.mode == "auto":
        return "rlm"
    return args.mode


def _read_inputs(args: argparse.Namespace, input_fields: list[str]) -> dict[str, Any]:
    if args.inputs_json and args.inputs_file:
        raise ValueError("Use only one of --inputs-json or --inputs-file.")

    if args.inputs_file:
        with open(args.inputs_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("--inputs-file must contain a JSON object.")
        return data

    if args.inputs_json:
        data = json.loads(args.inputs_json)
        if not isinstance(data, dict):
            raise ValueError("--inputs-json must be a JSON object.")
        return data

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if prompt is None:
        raise ValueError("Provide a prompt argument (or stdin), or pass --inputs-json/--inputs-file.")

    if len(input_fields) != 1:
        raise ValueError(
            "Prompt shorthand only works for single-input signatures. "
            "For multi-input signatures, pass --inputs-json/--inputs-file."
        )
    return {input_fields[0]: prompt}


def _pick_output_field(args: argparse.Namespace, output_fields: list[str]) -> str:
    if args.output_field:
        if args.output_field not in output_fields:
            raise ValueError(f"--output-field '{args.output_field}' is not one of: {output_fields}")
        return args.output_field

    if "answer" in output_fields:
        return "answer"
    return output_fields[0]


def _usage_summary(usage: dict[str, Any]) -> str:
    return (
        f"prompt_tokens={usage.get('prompt_tokens', 0)} "
        f"completion_tokens={usage.get('completion_tokens', 0)} "
        f"total_tokens={usage.get('total_tokens', 0)}"
    )


def _build_chat_query(history: list[tuple[str, str]], user_input: str) -> str:
    lines: list[str] = []
    for user_msg, assistant_msg in history:
        lines.append(f"USER:\n{user_msg}")
        lines.append(f"ASSISTANT:\n{assistant_msg}")
    lines.append(f"USER:\n{user_input}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _build_chat_messages(history: list[tuple[str, str]], user_input: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_input})
    return messages


def _extract_text_output(outputs: list[Any]) -> str:
    if not outputs:
        return ""

    first = outputs[0]
    if isinstance(first, dict) and "text" in first:
        return str(first["text"])
    return str(first)


def _startup_notices(args: argparse.Namespace, mode: Literal["rlm", "chat"]) -> list[str]:
    notices = [
        f"Session: mode={mode} theme={args.tui_theme} density={args.tui_density}",
    ]
    if mode == "rlm":
        notices.append("File access: ENABLED (local Python interpreter can read workspace files).")
        notices.append(f"Workspace cwd: {args.cwd}")
        notices.append("Visibility: /stats for last run metrics, /trace for recent trajectory snippets.")
    else:
        notices.append("File access: DISABLED in chat mode (text-only payload).")
        notices.append("Switch to --mode rlm for direct local file access.")
        notices.append("Visibility: /stats shows call metrics (trajectory is unavailable in chat mode).")
    return notices


def _history_count(lm: Any) -> int:
    history = getattr(lm, "history", None)
    if not isinstance(history, list):
        return 0
    return len(history)


def _extract_trajectory(prediction: Any) -> list[dict[str, Any]]:
    trajectory = getattr(prediction, "trajectory", None)
    if not isinstance(trajectory, list):
        return []
    result: list[dict[str, Any]] = []
    for step in trajectory:
        if isinstance(step, dict):
            result.append(step)
    return result


def _estimate_subquery_intents(trajectory: list[dict[str, Any]]) -> int:
    pattern = re.compile(r"\bllm_query(?:_batched|_with_media)?\s*\(")
    count = 0
    for step in trajectory:
        code = step.get("code", "")
        if isinstance(code, str):
            count += len(pattern.findall(code))
    return count


def _build_rlm_run_stats(prediction: Any, lm_calls_delta: int, max_depth: int) -> str:
    trajectory = _extract_trajectory(prediction)
    steps = len(trajectory)
    subqueries = _estimate_subquery_intents(trajectory)
    recursion = "on" if max_depth > 1 else "off"
    depth_budget = max(0, max_depth - 1)
    return (
        f"Run stats: steps={steps} lm_calls={lm_calls_delta} "
        f"subquery_intents={subqueries} recursion={recursion} depth_budget={depth_budget}"
    )


def _build_chat_run_stats(lm_calls_delta: int) -> str:
    return f"Run stats: lm_calls={lm_calls_delta} mode=chat (no trajectory)"


def _metrics_badge_from_stats(stats: str, mode: Literal["rlm", "chat"]) -> str:
    if not stats or stats.startswith("No run"):
        return "metrics: --"

    def _int_value(name: str) -> int | None:
        match = re.search(rf"{name}=([0-9]+)", stats)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    calls = _int_value("lm_calls")
    if mode == "chat":
        calls_part = f"calls:{calls}" if calls is not None else "calls:?"
        return calls_part

    steps = _int_value("steps")
    subq = _int_value("subquery_intents")
    depth = _int_value("depth_budget")

    parts = []
    parts.append(f"steps:{steps if steps is not None else '?'}")
    parts.append(f"calls:{calls if calls is not None else '?'}")
    parts.append(f"subq:{subq if subq is not None else '?'}")
    parts.append(f"depth:{depth if depth is not None else '?'}")
    return " ".join(parts)


def _preview_inline(text: Any, max_chars: int = 110) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def _trajectory_preview_lines(prediction: Any, max_steps: int = 5) -> list[str]:
    trajectory = _extract_trajectory(prediction)
    if not trajectory:
        return ["No trajectory is available yet."]

    lines: list[str] = []
    total = len(trajectory)
    shown = trajectory[-max_steps:]
    lines.append(f"Trajectory: total_steps={total}, showing_last={len(shown)}")
    start_index = total - len(shown) + 1
    for idx, step in enumerate(shown, start=start_index):
        code_preview = _preview_inline(step.get("code", ""), max_chars=70)
        out_preview = _preview_inline(step.get("output", ""), max_chars=100)
        lines.append(f"Step {idx}: code={code_preview} | out={out_preview}")
    return lines


def _trajectory_full_lines(prediction: Any, max_steps: int | None = None) -> list[str]:
    trajectory = _extract_trajectory(prediction)
    if not trajectory:
        return ["No trajectory is available yet."]

    total = len(trajectory)
    steps = trajectory if max_steps is None else trajectory[-max_steps:]
    start_index = total - len(steps) + 1

    lines: list[str] = []
    lines.append(f"Trajectory: total_steps={total} showing={len(steps)}")
    lines.append("Tip: use tmux copy-mode to copy blocks cleanly.")
    for idx, step in enumerate(steps, start=start_index):
        lines.append("")
        lines.append(f"Step {idx}/{total}")
        code = step.get("code", "")
        out = step.get("output", "")
        lines.append("code:")
        for ln in str(code).splitlines() or [""]:
            lines.append(f"  {ln}")
        lines.append("output:")
        for ln in str(out).splitlines() or [""]:
            lines.append(f"  {ln}")
    return lines


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _local_hhmm() -> str:
    """Local time for lightweight UI activity stamps."""
    try:
        return datetime.now().strftime("%H:%M")
    except Exception:
        return "--:--"


def _record_activity(session_state: dict[str, Any], text: str, limit: int = 8) -> None:
    """Append a short activity entry to session_state['activity'] (best-effort)."""
    if not isinstance(text, str) or not text.strip():
        return
    items = session_state.get("activity")
    if not isinstance(items, list):
        items = []
        session_state["activity"] = items
    entry = f"[{_local_hhmm()}] {text.strip()}"
    items.append(entry)
    if limit > 0 and len(items) > limit:
        del items[:-limit]


def _state_dir() -> Path:
    """Directory for persistent RLM CLI state (history, sessions)."""
    env = os.environ.get("DSPY_RLM_HOME") or os.environ.get("RLM_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".dspy" / "rlm"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_subdir(name: str) -> Path:
    """Best-effort state subdir, falling back to the system temp dir."""
    primary = _state_dir() / name
    try:
        _ensure_dir(primary)
        return primary
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "dspy_rlm" / name
        _ensure_dir(fallback)
        return fallback


def _sessions_dir() -> Path:
    return _safe_subdir("sessions")


def _history_path(mode_label: str) -> Path:
    d = _safe_subdir("history")
    return d / f"input_{mode_label}.jsonl"


def _new_session_id(prefix: str = "rlm") -> str:
    # Timestamp-based, filesystem-friendly id.
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{int(time.time() * 1000) % 100000}"


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Best-effort clipboard copy; returns (ok, provider_name)."""
    if not isinstance(text, str) or not text:
        return False, "empty"

    candidates: list[list[str]] = []
    if sys.platform == "darwin":
        candidates.append(["pbcopy"])
    candidates.append(["wl-copy"])
    candidates.append(["xclip", "-selection", "clipboard"])
    if os.name == "nt":
        candidates.append(["clip"])

    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            proc = subprocess.run(cmd, input=text, text=True, capture_output=True, check=False)
        except Exception:
            continue
        if proc.returncode == 0:
            return True, cmd[0]
    return False, "unavailable"


def _choose_editor() -> list[str]:
    value = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if value:
        try:
            parts = shlex.split(value)
            if parts:
                return parts
        except ValueError:
            pass
    if os.name == "nt":
        return ["notepad"]
    return ["vi"]


def _edit_text_in_editor(initial_text: str) -> str:
    editor_cmd = _choose_editor()
    tmp_dir: str | None = None
    try:
        tmp_dir = str(_state_dir())
        _ensure_dir(Path(tmp_dir))
    except Exception:
        tmp_dir = None
    with tempfile.NamedTemporaryFile(prefix="rlm_draft_", suffix=".txt", delete=False, dir=tmp_dir) as f:
        draft_path = Path(f.name)
        f.write((initial_text or "").encode("utf-8", errors="replace"))
        f.flush()

    try:
        subprocess.run([*editor_cmd, str(draft_path)], check=False)
        return draft_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            draft_path.unlink(missing_ok=True)
        except Exception:
            pass


def _edit_text_in_editor_from_curses(
    stdscr: Any,
    initial_text: str,
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
) -> str:
    """Run $EDITOR safely from inside curses and return edited text."""
    try:
        curses.def_prog_mode()
    except Exception:
        pass
    try:
        curses.endwin()
    except Exception:
        pass
    text = initial_text
    try:
        text = _edit_text_in_editor(initial_text)
    except Exception:
        text = initial_text
    finally:
        try:
            curses.reset_prog_mode()
        except Exception:
            pass
        try:
            stdscr.keypad(True)
        except Exception:
            pass
        _init_colors(theme)
        try:
            curses.curs_set(1)
        except Exception:
            pass
    return text


_SPINNER_FRAMES = ["-", "\\", "|", "/"]
_COLOR_HEADER = 1
_COLOR_STATUS = 2
_COLOR_USER = 3
_COLOR_ASSISTANT = 4
_COLOR_ERROR = 5
_COLOR_SYSTEM = 6
_COLOR_DIVIDER = 7
_COLOR_SURFACE = 8
_COLOR_SHADOW = 9
_COLOR_HEADER_SURF = 10
_COLOR_SYSTEM_SURF = 11
_COLOR_ASSISTANT_SURF = 12
_COLOR_DIM_SURF = 13
_THEME_PALETTES: dict[str, dict[str, tuple[int, int]]] = {
    "codex": {
        "header": (curses.COLOR_CYAN, -1),
        "status": (curses.COLOR_WHITE, -1),
        "user": (curses.COLOR_CYAN, -1),
        "assistant": (curses.COLOR_GREEN, -1),
        "error": (curses.COLOR_RED, -1),
        "system": (curses.COLOR_WHITE, -1),
        "divider": (curses.COLOR_WHITE, -1),
    },
    "neo": {
        "header": (curses.COLOR_BLACK, curses.COLOR_CYAN),
        "status": (curses.COLOR_BLACK, curses.COLOR_WHITE),
        "user": (curses.COLOR_CYAN, -1),
        "assistant": (curses.COLOR_GREEN, -1),
        "error": (curses.COLOR_RED, -1),
        "system": (curses.COLOR_YELLOW, -1),
        "divider": (curses.COLOR_BLUE, -1),
    },
    "amber": {
        "header": (curses.COLOR_BLACK, curses.COLOR_YELLOW),
        "status": (curses.COLOR_BLACK, curses.COLOR_MAGENTA),
        "user": (curses.COLOR_YELLOW, -1),
        "assistant": (curses.COLOR_WHITE, -1),
        "error": (curses.COLOR_RED, -1),
        "system": (curses.COLOR_MAGENTA, -1),
        "divider": (curses.COLOR_YELLOW, -1),
    },
    "mono": {
        "header": (curses.COLOR_WHITE, curses.COLOR_BLACK),
        "status": (curses.COLOR_BLACK, curses.COLOR_WHITE),
        "user": (curses.COLOR_WHITE, -1),
        "assistant": (curses.COLOR_WHITE, -1),
        "error": (curses.COLOR_WHITE, -1),
        "system": (curses.COLOR_WHITE, -1),
        "divider": (curses.COLOR_WHITE, -1),
    },
    "cyber": {
        # 256-color palette (slate borders + electric cyan accent). Falls back in _init_colors().
        "header": (45, -1),      # softer cyan
        "status": (245, -1),     # dim gray
        "user": (45, -1),        # cyan
        "assistant": (231, -1),  # crisp white
        "error": (203, -1),      # soft red
        "system": (252, -1),     # light gray
        "divider": (240, -1),    # slate gray (borders)
        # Depth: card surface + shadow use background colors to create "floating cards".
        "surface": (252, 235),   # light gray on dark slate
        "shadow": (0, 233),      # near-black shadow
        # Text styles that preserve the surface background.
        "header_surface": (45, 235),
        "system_surface": (252, 235),
        "assistant_surface": (231, 235),
        "dim_surface": (245, 235),
    },
}


class _LineEditor:
    """Single-line editor with cursor navigation and persistent history.

    Notes:
    - We support pasting multi-line content by storing real `\\n` in the buffer,
      but rendering it as the ASCII escape `\\n` to keep the input area one line.
    """

    def __init__(self, history_limit: int = 100, history_path: Path | None = None):
        self._buf: list[str] = []
        self._cursor = 0
        self._history: list[str] = []
        self._history_limit = history_limit
        self._history_path = history_path
        self._history_index: int | None = None
        self._saved_before_history: list[str] = []
        self._scroll_x = 0
        self._in_paste = False

        self._load_history()

    def _load_history(self) -> None:
        path = self._history_path
        if path is None or not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        # Backwards compat: treat the whole line as the entry.
                        self._history.append(raw)
                        continue
                    text = obj.get("text") if isinstance(obj, dict) else None
                    if isinstance(text, str) and text:
                        self._history.append(text)
        except Exception:
            return
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def clear(self) -> None:
        self._buf.clear()
        self._cursor = 0
        self._history_index = None
        self._saved_before_history = []
        self._scroll_x = 0
        self._in_paste = False

    def get_text(self) -> str:
        return "".join(self._buf)

    def set_text(self, text: str) -> None:
        self._set_text(text)

    def start_paste(self) -> None:
        self._in_paste = True

    def end_paste(self) -> None:
        self._in_paste = False

    def push_history(self, line: str) -> None:
        if not line or not line.strip():
            return
        if self._history and self._history[-1] == line:
            return
        self._history.append(line)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        if self._history_path is not None:
            try:
                _ensure_dir(self._history_path.parent)
                with self._history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _set_text(self, text: str) -> None:
        self._buf = list(text)
        self._cursor = len(self._buf)
        self._scroll_x = 0

    def _delete_word_backward(self) -> None:
        while self._cursor > 0 and self._buf[self._cursor - 1].isspace():
            self._buf.pop(self._cursor - 1)
            self._cursor -= 1
        while self._cursor > 0 and not self._buf[self._cursor - 1].isspace():
            self._buf.pop(self._cursor - 1)
            self._cursor -= 1

    def feed(self, ch: int) -> str | None:
        if ch in (10, 13, curses.KEY_ENTER):
            if self._in_paste:
                # Newlines inside bracketed paste should be inserted, not submitted.
                self._buf.insert(self._cursor, "\n")
                self._cursor += 1
                return None

            text = "".join(self._buf).rstrip("\n")
            self.clear()
            if text.strip():
                self.push_history(text)
                return text
            return None

        if ch == 3:  # Ctrl+C (clear current input)
            self.clear()
            return ""

        # Cursor movement
        if ch in (curses.KEY_LEFT, 2):  # Ctrl+B
            self._cursor = max(0, self._cursor - 1)
            return None
        if ch in (curses.KEY_RIGHT, 6):  # Ctrl+F
            self._cursor = min(len(self._buf), self._cursor + 1)
            return None
        if ch in (curses.KEY_HOME, 1):  # Ctrl+A
            self._cursor = 0
            return None
        if ch in (curses.KEY_END, 5):  # Ctrl+E
            self._cursor = len(self._buf)
            return None

        # History
        if ch == curses.KEY_UP:
            if not self._history:
                return None
            if self._history_index is None:
                self._saved_before_history = self._buf[:]
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self._set_text(self._history[self._history_index])
            return None

        if ch == curses.KEY_DOWN:
            if self._history_index is None:
                return None
            self._history_index += 1
            if self._history_index >= len(self._history):
                self._history_index = None
                self._buf = self._saved_before_history[:]
                self._cursor = len(self._buf)
                self._scroll_x = 0
            else:
                self._set_text(self._history[self._history_index])
            return None

        # Deletion
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if self._cursor > 0:
                self._buf.pop(self._cursor - 1)
                self._cursor -= 1
            return None
        if ch in (curses.KEY_DC, 4):  # Ctrl+D
            if self._cursor < len(self._buf):
                self._buf.pop(self._cursor)
            return None
        if ch == 11:  # Ctrl+K
            del self._buf[self._cursor:]
            return None
        if ch == 21:  # Ctrl+U
            del self._buf[:self._cursor]
            self._cursor = 0
            return None
        if ch == 23:  # Ctrl+W
            self._delete_word_backward()
            return None

        if ch == 9:  # Tab
            self._buf.insert(self._cursor, "\t")
            self._cursor += 1
            return None

        # Printable chars
        if 32 <= ch <= 0x10FFFF:
            self._buf.insert(self._cursor, chr(ch))
            self._cursor += 1
            return None

        return None

    def _display_text_and_cursor(self) -> tuple[str, int]:
        # Map embedded newlines/tabs to visible ASCII escapes so the input row doesn't break layout.
        out: list[str] = []
        cursor_x = 0
        for i, ch in enumerate(self._buf):
            if i == self._cursor:
                cursor_x = len(out)
            if ch == "\n":
                out.extend(["\\", "n"])
            elif ch == "\t":
                out.extend(["\\", "t"])
            elif ch == "\r":
                out.extend(["\\", "r"])
            else:
                out.append(ch)
        if self._cursor == len(self._buf):
            cursor_x = len(out)
        return "".join(out), cursor_x

    def get_display(self, max_width: int) -> tuple[str, int]:
        text, display_cursor = self._display_text_and_cursor()
        if max_width <= 0:
            return "", 0
        if len(text) <= max_width:
            self._scroll_x = 0
            return text, display_cursor

        if display_cursor < self._scroll_x:
            self._scroll_x = display_cursor
        elif display_cursor > self._scroll_x + max_width - 1:
            self._scroll_x = display_cursor - max_width + 1

        visible = text[self._scroll_x:self._scroll_x + max_width]
        cursor_x = max(0, min(len(visible), display_cursor - self._scroll_x))
        return visible, cursor_x


def _safe_addnstr(stdscr: Any, y: int, x: int, text: str, max_chars: int, attr: int = 0) -> None:
    if max_chars <= 0:
        return
    try:
        stdscr.addnstr(y, x, text, max_chars, attr)
    except curses.error:
        # Curses raises on narrow terminals or clipped writes; ignore and keep rendering.
        return


def _init_colors(theme: str = "codex") -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass

    # Graceful fallback: the cyber theme relies on 256-color indices.
    colors = int(getattr(curses, "COLORS", 0) or 0)
    if theme == "cyber" and colors and colors < 256:
        theme = "codex"

    palette = _THEME_PALETTES.get(theme, _THEME_PALETTES["neo"])

    # init_pair can raise on terminals with limited color support; keep rendering even if colors fail.
    try:
        curses.init_pair(_COLOR_HEADER, *palette["header"])
        curses.init_pair(_COLOR_STATUS, *palette["status"])
        curses.init_pair(_COLOR_USER, *palette["user"])
        curses.init_pair(_COLOR_ASSISTANT, *palette["assistant"])
        curses.init_pair(_COLOR_ERROR, *palette["error"])
        curses.init_pair(_COLOR_SYSTEM, *palette["system"])
        curses.init_pair(_COLOR_DIVIDER, *palette["divider"])
        extra_pairs = {
            "surface": _COLOR_SURFACE,
            "shadow": _COLOR_SHADOW,
            "header_surface": _COLOR_HEADER_SURF,
            "system_surface": _COLOR_SYSTEM_SURF,
            "assistant_surface": _COLOR_ASSISTANT_SURF,
            "dim_surface": _COLOR_DIM_SURF,
        }
        for key, pair_id in extra_pairs.items():
            if key in palette:
                curses.init_pair(pair_id, *palette[key])
    except curses.error:
        return


def _unicode_locale_likely() -> bool:
    """Best-effort check for UTF-8 locale/encoding (for box drawing)."""
    for key in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = os.environ.get(key, "")
        if "UTF-8" in value.upper() or "UTF8" in value.upper():
            return True
    enc = getattr(sys.stdout, "encoding", None) or ""
    if "UTF" in enc.upper():
        return True
    try:
        enc2 = locale.getpreferredencoding(False) or ""
    except Exception:
        enc2 = ""
    return "UTF" in enc2.upper()


def _resolve_tui_charset(value: str) -> Literal["ascii", "unicode"]:
    if value == "ascii":
        return "ascii"
    if value == "unicode":
        return "unicode"
    # auto
    return "unicode" if _unicode_locale_likely() else "ascii"


_BOX_CHARS: dict[str, dict[str, str]] = {
    "ascii": {"h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+"},
    # Rounded single-line box drawing characters.
    "unicode": {"h": "─", "v": "│", "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯"},
}


def _color_attr(pair_id: int) -> int:
    try:
        if not curses.has_colors():
            return 0
        return curses.color_pair(pair_id)
    except curses.error:
        return 0


def _is_codex_theme(theme: str) -> bool:
    return theme == "codex"


def _header_attr(theme: str) -> int:
    attr = _color_attr(_COLOR_HEADER) | curses.A_BOLD
    if _is_codex_theme(theme):
        attr |= curses.A_DIM
    return attr


def _status_attr(theme: str) -> int:
    attr = _color_attr(_COLOR_STATUS)
    if theme in {"codex", "cyber"}:
        attr |= curses.A_DIM
    return attr


def _badge_attr(theme: str) -> int:
    if _is_codex_theme(theme):
        return _color_attr(_COLOR_ASSISTANT) | curses.A_BOLD
    return _color_attr(_COLOR_HEADER) | curses.A_BOLD


def _role_tag(role: str) -> str:
    normalized = role.lower()
    if normalized == "you":
        return "YOU"
    if normalized == "error":
        return "ERROR"
    if normalized == "system":
        return "SYSTEM"
    if normalized == "codex":
        return "CODEX"
    return "RLM"


def _role_attr(role: str) -> int:
    normalized = role.lower()
    if normalized == "you":
        return _color_attr(_COLOR_USER)
    if normalized == "error":
        return _color_attr(_COLOR_ERROR)
    if normalized == "system":
        return _color_attr(_COLOR_SYSTEM)
    return _color_attr(_COLOR_ASSISTANT)


def _render_transcript_lines(
    transcript: list[tuple[str, str]],
    width: int,
    density: Literal["compact", "cozy"] = "compact",
    theme: Literal["codex", "neo", "amber", "mono", "cyber"] = "codex",
) -> list[tuple[str, int]]:
    lines: list[tuple[str, int]] = []
    if width < 20:
        return [("Window too narrow. Increase terminal width.", _color_attr(_COLOR_ERROR))]

    for role, text in transcript:
        tag = f"[{_role_tag(role)}]"
        wrap_width = max(10, width - len(tag) - 1)
        logical_lines = (text or "").splitlines() or [""]
        first_chunk = True
        for logical_line in logical_lines:
            wrapped = textwrap.wrap(
                logical_line,
                width=wrap_width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if not wrapped:
                wrapped = [""]

            for chunk in wrapped:
                if first_chunk:
                    lines.append((f"{tag} {chunk}", _role_attr(role)))
                    first_chunk = False
                else:
                    lines.append((" " * (len(tag) + 1) + chunk, _role_attr(role)))
        if _is_codex_theme(theme):
            separator = "-" * max(3, min(width - 1, 18))
        else:
            separator = "-" * max(1, min(width - 1, 48 if density == "compact" else 32))
        lines.append((separator, _color_attr(_COLOR_DIVIDER) | curses.A_DIM))
        if density == "cozy":
            lines.append(("", 0))

    return lines


def _render_transcript_lines_premium(
    transcript: list[tuple[str, str]],
    width: int,
    density: Literal["compact", "cozy"] = "compact",
) -> list[tuple[str, int]]:
    """Minimal transcript style (no noisy separators), closer to Codex/Claude CLIs."""
    lines: list[tuple[str, int]] = []
    if width < 20:
        return [("Window too narrow. Increase terminal width.", _color_attr(_COLOR_ERROR))]

    def _emit(prefix: str, text: str, attr: int) -> None:
        """Emit one message block with minimal formatting and codeblock preservation."""
        wrap_width = max(10, width - len(prefix))
        logical_lines = (text or "").splitlines() or [""]
        first = True
        in_code = False
        code_prefix = (" " * len(prefix)) + "| "
        for logical_line in logical_lines:
            stripped = logical_line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                if first:
                    lines.append((f"{prefix}{stripped}", attr | curses.A_DIM))
                    first = False
                else:
                    lines.append(((" " * len(prefix)) + stripped, attr | curses.A_DIM))
                continue

            if in_code:
                # Preserve code blocks without wrapping; clip at draw time.
                lines.append((f"{code_prefix}{logical_line}", attr))
                first = False
                continue

            wrapped = textwrap.wrap(
                logical_line,
                width=wrap_width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if not wrapped:
                wrapped = [""]
            for chunk in wrapped:
                if first:
                    lines.append((f"{prefix}{chunk}", attr))
                    first = False
                else:
                    lines.append(((" " * len(prefix)) + chunk, attr))

        # Spacing between message blocks.
        lines.append(("", 0))
        if density == "cozy":
            lines.append(("", 0))

    for role, text in transcript:
        normalized = role.lower()
        if normalized in {"you"}:
            _emit("> ", text, _role_attr(role) | curses.A_BOLD)
        elif normalized in {"rlm", "codex"}:
            _emit("  ", text, _role_attr(role))
        elif normalized == "error":
            _emit("! ", text, _role_attr(role) | curses.A_BOLD)
        else:
            # system/other
            _emit("  ", text, _role_attr("system") | curses.A_DIM)

    # Trim trailing blank spacer lines.
    while lines and not lines[-1][0].strip():
        lines.pop()
    return lines


def _premium_header_height(style: Literal["claude", "codex"]) -> int:
    # Claude: taller, two-panel header. Codex: compact floating box + spacer.
    return 9 if style == "claude" else 6


_BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
_BRACKETED_PASTE_DISABLE = "\x1b[?2004l"


def _set_bracketed_paste(enabled: bool) -> None:
    """Enable/disable terminal bracketed-paste mode (tmux-safe)."""
    try:
        if not sys.stdout or not sys.stdout.isatty():
            return
        sys.stdout.write(_BRACKETED_PASTE_ENABLE if enabled else _BRACKETED_PASTE_DISABLE)
        sys.stdout.flush()
    except Exception:
        return


def _is_compact_layout(height: int, width: int) -> bool:
    return height < 12 or width < 48


def _is_premium_layout(height: int, width: int, style: str) -> bool:
    """Whether the terminal is large enough for the premium (card + spotlight) layout."""
    if style not in {"claude", "codex"}:
        return False
    if width < 54:
        return False
    if _is_compact_layout(height, width):
        return False
    header_h = _premium_header_height(style)  # type: ignore[arg-type]
    # Reserve: header + 4 transcript rows + spotlight composer (3) + status bar (1).
    return height >= header_h + 8


def _premium_input_width(width: int) -> int:
    """Visible input width for the premium spotlight composer (excluding prefix)."""
    pad = 1
    safe_w = max(1, width - 1)
    box_w = max(10, safe_w - (2 * pad))
    content_w = max(1, box_w - 4)  # borders + 1-char padding each side
    prefix_len = 2  # "> "
    return max(1, content_w - prefix_len)


def _transcript_body_height(height: int, width: int) -> int:
    if _is_compact_layout(height, width):
        return max(1, height - 2)
    body_top = 2
    sep_y = height - 3
    return max(1, sep_y - body_top)


def _slice_transcript_lines(
    lines: list[tuple[str, int]],
    body_height: int,
    scroll_offset: int,
) -> tuple[list[tuple[str, int]], int, int]:
    total = len(lines)
    max_scroll = max(0, total - body_height)
    clamped = min(max(scroll_offset, 0), max_scroll)
    end = total - clamped
    start = max(0, end - body_height)
    return lines[start:end], clamped, max_scroll


def _scroll_percent(scroll_offset: int, max_scroll: int) -> int:
    if max_scroll <= 0:
        return 0
    return int((max(0, min(scroll_offset, max_scroll)) / max_scroll) * 100)


def _status_line(
    status: str,
    mode_label: str,
    transcript_count: int,
    scroll_offset: int,
    max_scroll: int,
    width: int,
    metrics_badge: str,
) -> str:
    mode = "RLM" if mode_label == "rlm" else "CHAT"
    left = f" {mode} | {status}".strip()
    right = f"{metrics_badge} | msgs:{transcript_count} scroll:{_scroll_percent(scroll_offset, max_scroll)}%"
    if width <= 1:
        return left
    max_total = max(1, width - 1)
    if len(left) + 1 + len(right) > max_total:
        left = left[:max(1, max_total - len(right) - 1)]
    gap = max(1, max_total - len(left) - len(right))
    return f"{left}{' ' * gap}{right}"


def _draw_compact_screen(
    stdscr: Any,
    transcript: list[tuple[str, str]],
    status: str,
    mode_label: str,
    input_text: str,
    input_cursor: int,
    scroll_offset: int,
    density: Literal["compact", "cozy"],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    metrics_badge: str,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    lines = _render_transcript_lines(transcript, max(1, width - 1), density=density, theme=theme)
    body_height = _transcript_body_height(height, width)
    visible, clamped_scroll, max_scroll = _slice_transcript_lines(lines, body_height, scroll_offset)

    for i, (line, attr) in enumerate(visible):
        _safe_addnstr(stdscr, i, 0, line, width - 1, attr)

    if clamped_scroll > 0:
        _safe_addnstr(stdscr, 0, max(0, width - 7), "^MORE", min(5, width - 1), _color_attr(_COLOR_SYSTEM) | curses.A_BOLD)
    if max_scroll > clamped_scroll:
        _safe_addnstr(
            stdscr,
            body_height - 1,
            max(0, width - 7),
            "vMORE",
            min(5, width - 1),
            _color_attr(_COLOR_SYSTEM) | curses.A_BOLD,
        )

    footer = _status_line(
        status,
        mode_label,
        len(transcript),
        clamped_scroll,
        max_scroll,
        width,
        metrics_badge,
    )
    _safe_addnstr(stdscr, height - 2, 0, footer, width - 1, _status_attr(theme))
    prefix = "you> "
    _safe_addnstr(stdscr, height - 1, 0, prefix, len(prefix), curses.A_BOLD)
    _safe_addnstr(stdscr, height - 1, len(prefix), input_text, max(1, width - len(prefix) - 1))
    try:
        stdscr.move(height - 1, min(width - 1, len(prefix) + input_cursor))
    except curses.error:
        pass
    stdscr.refresh()


def _draw_screen(
    stdscr: Any,
    transcript: list[tuple[str, str]],
    status: str,
    mode_label: str,
    input_text: str,
    input_cursor: int,
    scroll_offset: int,
    density: Literal["compact", "cozy"],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    metrics_badge: str,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    if _is_compact_layout(height, width):
        _draw_compact_screen(
            stdscr,
            transcript,
            status,
            mode_label,
            input_text,
            input_cursor,
            scroll_offset,
            density,
            theme,
            metrics_badge,
        )
        return

    left = f" {mode_label.upper()} | {theme} | {density} "
    right = " /help /edit /trace /stats /clear /quit "
    gap = max(1, width - len(left) - len(right) - 1)
    header_text = f"{left}{' ' * gap}{right}"
    _safe_addnstr(
        stdscr,
        0,
        0,
        header_text.ljust(width - 1),
        width - 1,
        _header_attr(theme),
    )
    badge = f"[{metrics_badge}]"
    if width > len(badge) + 2:
        _safe_addnstr(
            stdscr,
            0,
            max(0, width - len(badge) - 1),
            badge,
            len(badge),
            _badge_attr(theme),
        )

    try:
        stdscr.hline(1, 0, curses.ACS_HLINE, width)
    except curses.error:
        pass

    body_top = 2
    sep_y = height - 3
    status_y = height - 2
    input_y = height - 1
    body_height = _transcript_body_height(height, width)
    lines = _render_transcript_lines(transcript, max(1, width - 1), density=density, theme=theme)
    visible, clamped_scroll, max_scroll = _slice_transcript_lines(lines, body_height, scroll_offset)
    for idx, (line, attr) in enumerate(visible):
        _safe_addnstr(stdscr, body_top + idx, 0, line, width - 1, attr)

    if clamped_scroll > 0:
        _safe_addnstr(stdscr, body_top, max(0, width - 7), "^MORE", min(5, width - 1), _color_attr(_COLOR_SYSTEM) | curses.A_BOLD)
    if max_scroll > clamped_scroll:
        _safe_addnstr(
            stdscr,
            body_top + body_height - 1,
            max(0, width - 7),
            "vMORE",
            min(5, width - 1),
            _color_attr(_COLOR_SYSTEM) | curses.A_BOLD,
        )

    try:
        stdscr.hline(sep_y, 0, curses.ACS_HLINE, width)
    except curses.error:
        pass

    status_line = _status_line(
        status,
        mode_label,
        len(transcript),
        clamped_scroll,
        max_scroll,
        width,
        metrics_badge,
    )
    _safe_addnstr(
        stdscr,
        status_y,
        0,
        status_line.ljust(width - 1),
        width - 1,
        _status_attr(theme),
    )

    prompt = "you> "
    _safe_addnstr(stdscr, input_y, 0, prompt, len(prompt), curses.A_BOLD)
    _safe_addnstr(stdscr, input_y, len(prompt), input_text, max(1, width - len(prompt) - 1))
    try:
        stdscr.move(input_y, min(width - 1, len(prompt) + input_cursor))
    except curses.error:
        pass
    stdscr.refresh()


def _abbrev_middle(text: str, max_len: int) -> str:
    if not isinstance(text, str):
        text = str(text)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 8:
        return text[:max_len]
    keep_start = max(1, (max_len // 2) - 2)
    keep_end = max(1, max_len - keep_start - 3)
    return text[:keep_start] + "..." + text[-keep_end:]


def _pretty_path(path: str, max_len: int) -> str:
    if not path:
        return ""
    try:
        home = str(Path.home())
        real = os.path.realpath(path)
        if real.startswith(home + os.sep) or real == home:
            real = "~" + real[len(home):]
        return _abbrev_middle(real, max_len)
    except Exception:
        return _abbrev_middle(path, max_len)


def _draw_box(
    stdscr: Any,
    top: int,
    left: int,
    height: int,
    width: int,
    attr: int = 0,
    charset: Literal["ascii", "unicode"] = "ascii",
) -> None:
    if height < 2 or width < 2:
        return
    chars = _BOX_CHARS.get(charset, _BOX_CHARS["ascii"])
    try:
        for x in range(left + 1, left + width - 1):
            stdscr.addch(top, x, chars["h"], attr)
            stdscr.addch(top + height - 1, x, chars["h"], attr)
        for y in range(top + 1, top + height - 1):
            stdscr.addch(y, left, chars["v"], attr)
            stdscr.addch(y, left + width - 1, chars["v"], attr)
        stdscr.addch(top, left, chars["tl"], attr)
        stdscr.addch(top, left + width - 1, chars["tr"], attr)
        stdscr.addch(top + height - 1, left, chars["bl"], attr)
        stdscr.addch(top + height - 1, left + width - 1, chars["br"], attr)
    except Exception:
        # Wide-char rendering can fail on some terminals; fall back to ASCII borders.
        if charset != "ascii":
            try:
                _draw_box(stdscr, top, left, height, width, attr=attr, charset="ascii")
            except Exception:
                pass
        return


def _fill_rect(stdscr: Any, top: int, left: int, height: int, width: int, attr: int = 0) -> None:
    """Fill a rectangle with spaces (used for card surfaces)."""
    if height <= 0 or width <= 0:
        return
    line = " " * max(0, width)
    for y in range(top, top + height):
        _safe_addnstr(stdscr, y, left, line, width, attr)


def _fill_box_interior(stdscr: Any, top: int, left: int, height: int, width: int, attr: int = 0) -> None:
    if height < 3 or width < 3:
        return
    _fill_rect(stdscr, top + 1, left + 1, height - 2, width - 2, attr=attr)


def _draw_shadow(
    stdscr: Any,
    top: int,
    left: int,
    height: int,
    width: int,
    attr: int = 0,
    *,
    right: bool = True,
    bottom: bool = True,
) -> None:
    """Draw a subtle 1-col/1-row shadow to the right/bottom of a box."""
    if height < 2 or width < 2:
        return
    sx = left + width
    if right:
        # Right shadow column (exclude top border for a cleaner look).
        for y in range(top + 1, top + height):
            _safe_addnstr(stdscr, y, sx, " ", 1, attr)
    if bottom:
        # Bottom shadow row.
        _safe_addnstr(stdscr, top + height, left + 1, " " * width, width, attr)


def _draw_premium_screen(
    stdscr: Any,
    transcript: list[tuple[str, str]],
    status: str,
    mode_label: str,
    input_text: str,
    input_cursor: int,
    scroll_offset: int,
    density: Literal["compact", "cozy"],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    metrics_badge: str,
    style: Literal["claude", "codex"],
    model: str,
    reasoning_effort: str,
    cwd: str,
    session_id: str,
    autosave_path: str,
    context_enabled: bool,
    placeholder: str,
    activity: list[str],
    charset: Literal["ascii", "unicode"],
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    header_h = _premium_header_height(style)
    if not _is_premium_layout(height, width, style):
        # Too small for premium; fall back to classic screen.
        _draw_screen(
            stdscr,
            transcript,
            status,
            mode_label,
            input_text,
            input_cursor,
            scroll_offset,
            density,
            theme,
            metrics_badge,
        )
        return

    pad = 1
    safe_w = max(1, width - 1)  # avoid last column
    inner_w = max(20, safe_w - (2 * pad))
    box_attr = _color_attr(_COLOR_DIVIDER) | curses.A_DIM
    dim_attr = _color_attr(_COLOR_SYSTEM) | curses.A_DIM
    head_attr = _color_attr(_COLOR_HEADER) | curses.A_BOLD
    surface_attr = _color_attr(_COLOR_SURFACE)
    shadow_attr = _color_attr(_COLOR_SHADOW) | curses.A_DIM

    use_surface = theme == "cyber" and surface_attr != 0
    surf_head_attr = (_color_attr(_COLOR_HEADER_SURF) | curses.A_BOLD) if use_surface else head_attr
    surf_dim_attr = (_color_attr(_COLOR_DIM_SURF) | curses.A_DIM) if use_surface else dim_attr
    surf_sys_attr = _color_attr(_COLOR_SYSTEM_SURF) if use_surface else _color_attr(_COLOR_SYSTEM)
    surf_asst_attr = _color_attr(_COLOR_ASSISTANT_SURF) if use_surface else _role_attr("assistant")

    # Spotlight composer (3 lines) + status bar (1).
    footer_y = height - 1
    composer_top = height - 4
    composer_h = 3
    composer_input_y = composer_top + 1

    # Transcript region (compute early so header cards can show scroll percent).
    body_top = header_h
    body_height = max(1, composer_top - body_top)
    lines = _render_transcript_lines_premium(transcript, max(1, width - 2), density=density)
    visible, clamped_scroll, max_scroll = _slice_transcript_lines(lines, body_height, scroll_offset)
    scroll_pct = _scroll_percent(clamped_scroll, max_scroll)

    if style == "claude":
        # Claude style: floating cards (Profile + Quick Start).
        card_h = max(6, header_h - 1)  # keep one spacer row beneath
        gap = 2
        avail = max(30, inner_w)
        min_left = 28
        min_right = 24
        left_w = min(38, max(min_left, avail // 3))
        right_w = avail - left_w - gap
        if right_w < min_right:
            left_w = max(min_left, avail - gap - min_right)
            right_w = avail - left_w - gap
        if right_w < min_right:
            # Fallback to a single header box if the split doesn't fit.
            left_w = avail
            right_w = 0
            gap = 0

        left_x = pad
        right_x = pad + left_w + gap
        top_y = 0

        _draw_box(stdscr, top_y, left_x, card_h, left_w, box_attr, charset=charset)
        if right_w >= 10:
            _draw_box(stdscr, top_y, right_x, card_h, right_w, box_attr, charset=charset)
        if use_surface:
            _fill_box_interior(stdscr, top_y, left_x, card_h, left_w, attr=surface_attr)
            if shadow_attr:
                _draw_shadow(stdscr, top_y, left_x, card_h, left_w, attr=shadow_attr, right=True, bottom=True)
            if right_w >= 10:
                _fill_box_interior(stdscr, top_y, right_x, card_h, right_w, attr=surface_attr)
                if shadow_attr:
                    _draw_shadow(stdscr, top_y, right_x, card_h, right_w, attr=shadow_attr, right=True, bottom=True)

        _safe_addnstr(stdscr, top_y, left_x + 2, " Profile ", max(1, left_w - 4), surf_head_attr)
        if right_w >= 10:
            _safe_addnstr(stdscr, top_y, right_x + 2, " Quick Start ", max(1, right_w - 4), surf_head_attr)

        # Left card content.
        user = (os.environ.get("USER") or os.environ.get("USERNAME") or "").strip()
        if user and 1 <= len(user) <= 18 and user.isprintable():
            greet = f"Welcome back, {user}!"
        else:
            greet = "Welcome back."
        _safe_addnstr(stdscr, top_y + 1, left_x + 2, greet, max(1, left_w - 4), surf_sys_attr | curses.A_BOLD)

        bot = [
            " .----.",
            " |o  o|",
            " | -- |",
            " '----'",
        ]
        bot_w = max(len(ln) for ln in bot)
        bot_x = left_x + 2
        bot_y = top_y + 2
        for i, line in enumerate(bot):
            y = bot_y + i
            if y >= top_y + card_h - 2:
                break
            _safe_addnstr(stdscr, y, bot_x, line, max(1, left_w - 4), surf_asst_attr | curses.A_BOLD)

        info_x = bot_x + bot_w + 2
        info_w = max(1, (left_x + left_w - 2) - info_x)
        if info_w >= 10:
            _safe_addnstr(stdscr, bot_y + 0, info_x, "model: ", min(7, info_w), surf_dim_attr)
            _safe_addnstr(stdscr, bot_y + 0, info_x + 7, model, max(1, info_w - 7), surf_sys_attr)
            _safe_addnstr(stdscr, bot_y + 1, info_x, "tier:  ", min(7, info_w), surf_dim_attr)
            _safe_addnstr(stdscr, bot_y + 1, info_x + 7, reasoning_effort, max(1, info_w - 7), surf_sys_attr)
            _safe_addnstr(stdscr, bot_y + 2, info_x, "dir:   ", min(7, info_w), surf_dim_attr)
            _safe_addnstr(
                stdscr,
                bot_y + 2,
                info_x + 7,
                _pretty_path(cwd, max(10, info_w - 7)),
                max(1, info_w - 7),
                surf_sys_attr,
            )

        # Context + scroll gauge (bottom interior row).
        bar_w = 10
        filled = int((scroll_pct / 100) * bar_w) if scroll_pct > 0 else 0
        bar = "[" + ("#" * filled).ljust(bar_w, "-") + "]"
        ctx = "on" if context_enabled else "off"
        gauge = f"Ctx:{ctx}  Scr:{bar} {scroll_pct}%"
        _safe_addnstr(stdscr, top_y + card_h - 2, left_x + 2, gauge, max(1, left_w - 4), surf_dim_attr)

        # Right card content.
        if right_w >= 10:
            rw = max(1, right_w - 4)
            _safe_addnstr(stdscr, top_y + 1, right_x + 2, "> /help  commands", rw, surf_dim_attr)
            _safe_addnstr(stdscr, top_y + 2, right_x + 2, "> /edit  compose in $EDITOR", rw, surf_dim_attr)
            _safe_addnstr(stdscr, top_y + 3, right_x + 2, "> ^C     cancel request", rw, surf_dim_attr)
            _safe_addnstr(stdscr, top_y + 4, right_x + 2, "Recent activity", rw, surf_head_attr)
            recent = activity[-2:] if isinstance(activity, list) else []
            if not recent:
                recent = ["[--:--] (none)"]
            for i, entry in enumerate(recent[:2]):
                _safe_addnstr(stdscr, top_y + 5 + i, right_x + 2, f"- {entry}", rw, surf_dim_attr)
    else:
        # Codex style: a small floating header box (minimal, like codex CLI).
        box_h = header_h - 1  # leave a blank spacer row beneath
        box_w = min(inner_w, 54)
        box_w = max(34, box_w)
        _draw_box(stdscr, 0, pad, box_h, box_w, box_attr, charset=charset)
        if use_surface:
            _fill_box_interior(stdscr, 0, pad, box_h, box_w, attr=surface_attr)
            if shadow_attr:
                _draw_shadow(stdscr, 0, pad, box_h, box_w, attr=shadow_attr, right=True, bottom=True)

        app_label = "DSPy RLM" if mode_label == "rlm" else "DSPy Chat"
        title_line = f">- {app_label}"
        _safe_addnstr(stdscr, 1, pad + 2, title_line, max(1, box_w - 4), surf_head_attr if use_surface else _badge_attr(theme))
        _safe_addnstr(
            stdscr,
            2,
            pad + 2,
            f"model: {model}  {reasoning_effort}".strip(),
            max(1, box_w - 4),
            surf_dim_attr if use_surface else (_color_attr(_COLOR_SYSTEM) | curses.A_DIM),
        )
        _safe_addnstr(
            stdscr,
            3,
            pad + 2,
            f"directory: {_pretty_path(cwd, max(10, box_w - 14))}",
            max(1, box_w - 4),
            surf_dim_attr if use_surface else (_color_attr(_COLOR_SYSTEM) | curses.A_DIM),
        )

    for idx, (line, attr) in enumerate(visible):
        _safe_addnstr(stdscr, body_top + idx, pad, line, max(1, width - 3), attr)

    if not transcript:
        hint = "(chat will stream and scroll here)"
        y = body_top + max(0, body_height // 2)
        x = max(pad, (width - len(hint)) // 2)
        _safe_addnstr(stdscr, y, x, hint, max(1, width - x - 1), dim_attr)

    # Scroll affordances.
    if clamped_scroll > 0:
        _safe_addnstr(stdscr, body_top, max(0, width - 7), "^MORE", min(5, width - 1), _color_attr(_COLOR_SYSTEM) | curses.A_BOLD)
    if max_scroll > clamped_scroll:
        _safe_addnstr(
            stdscr,
            body_top + body_height - 1,
            max(0, width - 7),
            "vMORE",
            min(5, width - 1),
            _color_attr(_COLOR_SYSTEM) | curses.A_BOLD,
        )

    # Spotlight composer box.
    _draw_box(stdscr, composer_top, pad, composer_h, inner_w, box_attr, charset=charset)
    if use_surface:
        _fill_box_interior(stdscr, composer_top, pad, composer_h, inner_w, attr=surface_attr)
        if shadow_attr:
            _draw_shadow(stdscr, composer_top, pad, composer_h, inner_w, attr=shadow_attr, right=True, bottom=False)
    label = " Ask RLM " if mode_label == "rlm" else " Ask Codex "
    _safe_addnstr(stdscr, composer_top, pad + 2, label, max(1, inner_w - 4), surf_head_attr if use_surface else head_attr)

    composer_attr = _status_attr(theme) | curses.A_DIM
    _safe_addnstr(
        stdscr,
        composer_input_y,
        pad + 1,
        " " * max(1, inner_w - 2),
        max(1, inner_w - 2),
        surface_attr if use_surface else composer_attr,
    )

    prefix = "> "
    text_x = pad + 2 + len(prefix)
    _safe_addnstr(
        stdscr,
        composer_input_y,
        pad + 2,
        prefix,
        len(prefix),
        (surf_head_attr if use_surface else curses.A_BOLD),
    )

    # Hint + ghost completion (best-effort).
    hint = "[Tab] auto-complete"
    hint_x: int | None = None
    if inner_w >= len(hint) + 24:
        hint_x = pad + inner_w - 2 - len(hint)
        _safe_addnstr(stdscr, composer_input_y, hint_x, hint, len(hint), surf_dim_attr if use_surface else dim_attr)

    # Keep one column of right padding inside the box.
    text_end_x = pad + inner_w - 3
    if hint_x is not None:
        # Reserve hint space so long input doesn't overwrite it.
        text_end_x = min(text_end_x, hint_x - 2)
    text_max = max(1, text_end_x - text_x + 1)

    raw = input_text or ""
    ghost = ""
    if input_cursor == len(raw) and raw.startswith("/") and "\n" not in raw and " " not in raw:
        matches = [cmd for cmd in _TUI_COMMANDS if cmd.startswith(raw)]
        if matches:
            common = os.path.commonprefix(matches)
            if len(common) > len(raw):
                ghost = common[len(raw):]
            elif len(matches) == 1 and len(matches[0]) > len(raw):
                ghost = matches[0][len(raw):]

    if not raw:
        _safe_addnstr(
            stdscr,
            composer_input_y,
            text_x,
            placeholder,
            text_max,
            surf_dim_attr if use_surface else dim_attr,
        )
        try:
            stdscr.move(composer_input_y, min(text_end_x, text_x))
        except curses.error:
            pass
    else:
        _safe_addnstr(stdscr, composer_input_y, text_x, raw, text_max, surf_sys_attr if use_surface else 0)
        if ghost:
            ghost_x = text_x + len(raw)
            if ghost_x <= text_end_x:
                _safe_addnstr(
                    stdscr,
                    composer_input_y,
                    ghost_x,
                    ghost,
                    max(1, text_end_x - ghost_x + 1),
                    surf_dim_attr if use_surface else dim_attr,
                )
        try:
            cursor_x = min(text_end_x, text_x + input_cursor)
            stdscr.move(composer_input_y, cursor_x)
        except curses.error:
            pass

    # Footer (minimal).
    left = "^P Palette | ? Shortcuts"
    right = f"ctx:{'on' if context_enabled else 'off'} | {metrics_badge} | scr:{scroll_pct}%"
    center = status.strip()
    max_total = max(1, width - 1)
    left_part = f" {left} | "
    right_part = f" | {right} "
    # Allocate remaining space to center.
    remaining = max_total - len(left_part) - len(right_part)
    if remaining < 1:
        remaining = 1
        right_part = _abbrev_middle(right_part, max(1, max_total - len(left_part) - remaining))
    center_part = _abbrev_middle(center, remaining).ljust(remaining)
    footer = f"{left_part}{center_part}{right_part}"
    _safe_addnstr(stdscr, footer_y, 0, footer.ljust(max_total), max_total, _status_attr(theme))

    stdscr.refresh()


def _draw_main_screen(
    stdscr: Any,
    transcript: list[tuple[str, str]],
    status: str,
    mode_label: str,
    input_text: str,
    input_cursor: int,
    scroll_offset: int,
    density: Literal["compact", "cozy"],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    metrics_badge: str,
    style: Literal["claude", "codex", "classic"],
    model: str,
    reasoning_effort: str,
    cwd: str,
    session_id: str,
    autosave_path: str,
    context_enabled: bool,
    placeholder: str,
    activity: list[str],
    charset: Literal["ascii", "unicode"],
) -> None:
    if style in {"claude", "codex"}:
        _draw_premium_screen(
            stdscr,
            transcript,
            status,
            mode_label,
            input_text,
            input_cursor,
            scroll_offset,
            density,
            theme,
            metrics_badge,
            style,
            model,
            reasoning_effort,
            cwd,
            session_id,
            autosave_path,
            context_enabled,
            placeholder,
            activity,
            charset,
        )
    else:
        _draw_screen(
            stdscr,
            transcript,
            status,
            mode_label,
            input_text,
            input_cursor,
            scroll_offset,
            density,
            theme,
            metrics_badge,
        )


def _consume_bracketed_paste(stdscr: Any, restore_timeout: int) -> Literal["paste_start", "paste_end"] | None:
    """If an ESC sequence for bracketed paste is pending, consume it and return a token."""
    try:
        stdscr.timeout(0)
        ch1 = stdscr.getch()
        if ch1 == -1:
            return None
        if ch1 != ord("["):
            curses.ungetch(ch1)
            return None

        ch2 = stdscr.getch()
        ch3 = stdscr.getch()
        ch4 = stdscr.getch()
        ch5 = stdscr.getch()
        if -1 in (ch2, ch3, ch4, ch5):
            for ch in (ch5, ch4, ch3, ch2):
                if ch != -1:
                    curses.ungetch(ch)
            curses.ungetch(ch1)
            return None

        seq = "".join(chr(c) for c in (ch2, ch3, ch4, ch5))
        if seq == "200~":
            return "paste_start"
        if seq == "201~":
            return "paste_end"

        # Unknown escape: push back in reverse.
        for ch in (ch5, ch4, ch3, ch2, ch1):
            curses.ungetch(ch)
        return None
    finally:
        try:
            stdscr.timeout(restore_timeout)
        except curses.error:
            pass


def _boot_animation(
    stdscr: Any,
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    charset: Literal["ascii", "unicode"],
    duration_s: float = 0.9,
) -> None:
    """Short, skippable ASCII boot animation (tmux-safe)."""
    logo = [
        r"  ____  _     __  __",
        r" |  _ \| |   |  \/  |",
        r" | |_) | |   | |\/| |",
        r" |  _ <| |___| |  | |",
        r" |_| \_\_____|_|  |_|",
    ]
    try:
        height, width = stdscr.getmaxyx()
    except curses.error:
        return

    if height < 9 or width < 28:
        return

    start = time.monotonic()
    frames = max(6, min(18, int(duration_s / 0.08)))
    spinner = ["|", "/", "-", "\\"]
    stdscr.nodelay(True)
    try:
        for i in range(frames):
            if stdscr.getch() != -1:
                break
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= duration_s:
                break

            stdscr.erase()
            height, width = stdscr.getmaxyx()
            box_w = min(width - 2, max(len(line) for line in logo) + 4)
            box_h = min(height - 2, len(logo) + 4)
            top = max(0, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)

            _draw_box(
                stdscr,
                top,
                left,
                box_h,
                box_w,
                attr=_color_attr(_COLOR_DIVIDER) | curses.A_DIM,
                charset=charset,
            )

            scan_row = int((i / max(1, frames - 1)) * (len(logo) - 1))
            for row, line in enumerate(logo):
                y = top + 2 + row
                x = left + 2
                attr = curses.A_BOLD | _role_attr("assistant")
                if row == scan_row:
                    attr |= curses.A_REVERSE
                try:
                    stdscr.addnstr(y, x, line, box_w - 4, attr)
                except curses.error:
                    pass

            bar_w = max(10, min(18, box_w - 12))
            filled = int((i / max(1, frames - 1)) * bar_w)
            bar = "[" + ("#" * filled).ljust(bar_w, "-") + "]"
            msg = f"{bar} booting {spinner[i % len(spinner)]}"
            try:
                stdscr.addnstr(top + box_h - 2, left + 2, msg, box_w - 4, _status_attr(theme))
            except curses.error:
                pass

            stdscr.refresh()
            time.sleep(0.05)
    finally:
        stdscr.nodelay(False)
        try:
            stdscr.timeout(-1)
        except curses.error:
            pass


def _pager(
    stdscr: Any,
    title: str,
    lines: list[str],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
) -> None:
    """Simple scrollable pager. Quit with q or Esc."""
    scroll = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        header = f" {title}  (q/Esc to close, PgUp/PgDn scroll) "
        _safe_addnstr(stdscr, 0, 0, header.ljust(max(1, width - 1)), width - 1, _header_attr(theme))
        footer_y = max(0, height - 1)
        body_top = 1
        body_height = max(0, footer_y - body_top)

        wrap_width = max(10, width - 2)
        wrapped: list[str] = []
        for line in lines:
            if line is None:
                line = ""
            text = str(line).replace("\t", "    ")
            chunks = textwrap.wrap(text, width=wrap_width, replace_whitespace=False, drop_whitespace=False)
            wrapped.extend(chunks if chunks else [""])

        max_scroll = max(0, len(wrapped) - body_height)
        scroll = min(max(scroll, 0), max_scroll)

        visible = wrapped[scroll:scroll + body_height]
        for i, line in enumerate(visible):
            _safe_addnstr(stdscr, body_top + i, 0, line, max(1, width - 1))

        percent = _scroll_percent(scroll, max_scroll)
        footer = f" lines:{len(wrapped)} scroll:{percent}% "
        _safe_addnstr(stdscr, footer_y, 0, footer.ljust(max(1, width - 1)), width - 1, _status_attr(theme))
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), 27):
            break
        if ch == curses.KEY_RESIZE:
            continue
        if ch == curses.KEY_PPAGE:
            scroll = max(0, scroll - max(1, body_height // 2))
            continue
        if ch == curses.KEY_NPAGE:
            scroll = min(max_scroll, scroll + max(1, body_height // 2))
            continue
        if ch == curses.KEY_UP:
            scroll = max(0, scroll - 1)
            continue
        if ch == curses.KEY_DOWN:
            scroll = min(max_scroll, scroll + 1)
            continue
        if ch == curses.KEY_HOME:
            scroll = 0
            continue
        if ch == curses.KEY_END:
            scroll = max_scroll
            continue


_TUI_COMMANDS: list[str] = [
    "/help",
    "/quit",
    "/exit",
    "/clear",
    "/stats",
    "/trace",
    "/sessions",
    "/load",
    "/session",
    "/export",
    "/edit",
    "/retry",
    "/context",
    "/model",
    "/reasoning",
    "/theme",
    "/density",
    "/style",
    "/charset",
    "/copy",
]

_TUI_COMMANDS_WITH_ARGS: set[str] = {
    "/trace",
    "/load",
    "/export",
    "/edit",
    "/context",
    "/model",
    "/reasoning",
    "/theme",
    "/density",
    "/style",
    "/charset",
    "/copy",
}

_TUI_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/help": "Show commands and keybindings",
    "/quit": "Exit the TUI",
    "/exit": "Exit the TUI",
    "/clear": "Clear transcript",
    "/stats": "Show last run stats",
    "/trace": "View recent trajectory (RLM mode)",
    "/sessions": "List recent sessions",
    "/load": "Load a session by id or path",
    "/session": "Show current session details",
    "/export": "Export session as md or json",
    "/edit": "Compose in $EDITOR",
    "/retry": "Retry last user message",
    "/context": "Toggle conversation context on/off",
    "/model": "Set/inspect model",
    "/reasoning": "Set/inspect reasoning effort",
    "/theme": "Set TUI theme",
    "/density": "Set transcript density",
    "/style": "Set layout style",
    "/charset": "Set border charset (auto/ascii/unicode)",
    "/copy": "Copy last answer or full transcript",
}


def _rank_command_match(cmd: str, desc: str, query: str) -> tuple[int, int, str]:
    """Lower score is better."""
    q = (query or "").strip().lower()
    if not q:
        return (50, len(cmd), cmd)
    c = cmd.lower()
    d = desc.lower()
    q2 = q[1:] if q.startswith("/") else q
    c2 = c[1:] if c.startswith("/") else c

    if c == q or c2 == q2:
        return (0, len(cmd), cmd)
    if c.startswith(q) or (q2 and c2.startswith(q2)):
        return (1, len(cmd), cmd)
    if q in c or (q2 and q2 in c2):
        return (2, len(cmd), cmd)
    if q in d:
        return (3, len(cmd), cmd)
    return (99, len(cmd), cmd)


def _command_palette(
    stdscr: Any,
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    charset: Literal["ascii", "unicode"],
) -> str | None:
    """Spotlight-like command palette. Returns a command string (maybe with args space) or None."""
    query = ""
    selected = 0

    stdscr.timeout(-1)
    while True:
        height, width = stdscr.getmaxyx()
        pad = 2
        box_w = min(max(44, width - (2 * pad)), 78)
        box_h = min(max(9, height - (2 * pad)), 14)
        top = max(0, (height - box_h) // 2)
        left = max(0, (width - box_w) // 2)

        # Compute matches.
        items: list[str] = []
        for cmd in _TUI_COMMANDS:
            desc = _TUI_COMMAND_DESCRIPTIONS.get(cmd, "")
            score = _rank_command_match(cmd, desc, query)
            if score[0] < 99:
                items.append(cmd)
        items.sort(key=lambda c: _rank_command_match(c, _TUI_COMMAND_DESCRIPTIONS.get(c, ""), query))
        if not items:
            items = []
            selected = 0
        else:
            selected = max(0, min(selected, len(items) - 1))

        # Draw.
        stdscr.erase()
        box_attr = _color_attr(_COLOR_DIVIDER) | curses.A_DIM
        surface_attr = _color_attr(_COLOR_SURFACE)
        shadow_attr = _color_attr(_COLOR_SHADOW) | curses.A_DIM
        use_surface = theme == "cyber" and surface_attr != 0
        head_attr = (_color_attr(_COLOR_HEADER_SURF) | curses.A_BOLD) if use_surface else (_color_attr(_COLOR_HEADER) | curses.A_BOLD)
        dim_attr = (_color_attr(_COLOR_DIM_SURF) | curses.A_DIM) if use_surface else (_color_attr(_COLOR_SYSTEM) | curses.A_DIM)
        sys_attr = _color_attr(_COLOR_SYSTEM_SURF) if use_surface else 0
        cmd_attr_base = (_color_attr(_COLOR_HEADER_SURF) | curses.A_BOLD) if use_surface else (_role_attr("assistant") | curses.A_BOLD)

        _draw_box(stdscr, top, left, box_h, box_w, box_attr, charset=charset)
        if use_surface:
            _fill_box_interior(stdscr, top, left, box_h, box_w, attr=surface_attr)
            if shadow_attr:
                _draw_shadow(stdscr, top, left, box_h, box_w, attr=shadow_attr, right=True, bottom=True)
        _safe_addnstr(stdscr, top, left + 2, " Command Palette ", max(1, box_w - 4), head_attr)

        hint = "Esc close  Enter insert"
        if box_w >= len(hint) + 8:
            _safe_addnstr(stdscr, top, left + box_w - 2 - len(hint), hint, len(hint), dim_attr)

        # Query row.
        q_prefix = "> "
        q_y = top + 1
        q_x = left + 2
        q_w = max(1, box_w - 4)
        _safe_addnstr(stdscr, q_y, q_x, " " * q_w, q_w, surface_attr if use_surface else (_status_attr(theme) | curses.A_DIM))
        _safe_addnstr(stdscr, q_y, q_x, q_prefix, len(q_prefix), head_attr)
        _safe_addnstr(stdscr, q_y, q_x + len(q_prefix), query, max(1, q_w - len(q_prefix)), sys_attr)

        # Results.
        list_top = top + 3
        list_h = max(1, box_h - 5)
        view = items[:list_h]
        for i, cmd in enumerate(view):
            y = list_top + i
            desc = _TUI_COMMAND_DESCRIPTIONS.get(cmd, "")
            row_attr = curses.A_REVERSE if i == selected else 0
            cmd_attr = cmd_attr_base | row_attr
            desc_attr = dim_attr | row_attr
            _safe_addnstr(stdscr, y, left + 2, cmd.ljust(12), max(1, box_w - 4), cmd_attr)
            if desc:
                _safe_addnstr(stdscr, y, left + 2 + 13, desc, max(1, box_w - 4 - 13), desc_attr)

        if not items:
            _safe_addnstr(stdscr, list_top, left + 2, "(no matches)", max(1, box_w - 4), dim_attr)

        # Cursor.
        try:
            cur_x = min(left + box_w - 3, q_x + len(q_prefix) + len(query))
            stdscr.move(q_y, cur_x)
        except curses.error:
            pass

        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (27,):  # Esc
            return None
        if ch in (10, 13, curses.KEY_ENTER):
            if not items:
                continue
            cmd = items[selected]
            suffix = " " if cmd in _TUI_COMMANDS_WITH_ARGS else ""
            return cmd + suffix
        if ch in (curses.KEY_UP,):
            selected = max(0, selected - 1)
            continue
        if ch in (curses.KEY_DOWN,):
            selected = min(max(0, len(items) - 1), selected + 1)
            continue
        if ch == curses.KEY_PPAGE:
            selected = max(0, selected - max(1, list_h // 2))
            continue
        if ch == curses.KEY_NPAGE:
            selected = min(max(0, len(items) - 1), selected + max(1, list_h // 2))
            continue

        # Basic text input.
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            if query:
                query = query[:-1]
                selected = 0
            continue
        if 32 <= ch <= 0x10FFFF:
            query += chr(ch)
            selected = 0
            continue


def _handle_tab_completion(
    stdscr: Any,
    editor: _LineEditor,
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
) -> bool:
    """Return True if Tab was consumed for command completion."""
    text = editor.get_text()
    if not text.startswith("/") or "\n" in text or " " in text:
        return False

    matches = [cmd for cmd in _TUI_COMMANDS if cmd.startswith(text)]
    if not matches:
        try:
            curses.beep()
        except curses.error:
            pass
        return True

    if len(matches) == 1:
        cmd = matches[0]
        suffix = " " if cmd in _TUI_COMMANDS_WITH_ARGS else ""
        editor.set_text(cmd + suffix)
        return True

    common = os.path.commonprefix(matches)
    if common and len(common) > len(text):
        editor.set_text(common)
        return True

    _pager(
        stdscr,
        "Command matches",
        ["Multiple matches:", "", *matches, "", "Tip: type more letters and press Tab again."],
        theme,
    )
    return True


def _run_with_spinner(
    stdscr: Any,
    transcript: list[tuple[str, str]],
    mode_label: str,
    base_status: str,
    fn: Any,
    editor: _LineEditor,
    scroll_offset: int,
    density: Literal["compact", "cozy"],
    theme: Literal["codex", "neo", "amber", "mono", "cyber"],
    metrics_badge: str,
    progress_fn: Callable[[], str] | None = None,
    progress_count_fn: Callable[[], int] | None = None,
    stall_warn_seconds: int = 120,
    request_timeout_seconds: float | None = None,
    cancel_event: threading.Event | None = None,
    draw_context: dict[str, Any] | None = None,
) -> Any:
    outcome: dict[str, Any] = {}
    failure: dict[str, Exception] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            outcome["value"] = fn()
        except Exception as exc:
            failure["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    frame_idx = 0
    start_time = time.monotonic()
    last_progress_change_time = start_time
    last_progress_count = 0
    if progress_count_fn is not None:
        try:
            last_progress_count = max(0, int(progress_count_fn()))
        except Exception:
            last_progress_count = 0
    stdscr.timeout(130)
    try:
        while not done.is_set():
            frame = _SPINNER_FRAMES[frame_idx % len(_SPINNER_FRAMES)]
            elapsed = max(0, int(time.monotonic() - start_time))
            progress = ""
            if progress_fn is not None:
                try:
                    progress = progress_fn().strip()
                except Exception:
                    progress = ""
            current_progress_count: int | None = None
            if progress_count_fn is not None:
                try:
                    current_progress_count = max(0, int(progress_count_fn()))
                except Exception:
                    current_progress_count = None

            if current_progress_count is not None and current_progress_count != last_progress_count:
                last_progress_count = current_progress_count
                last_progress_change_time = time.monotonic()

            idle_seconds = max(0, int(time.monotonic() - last_progress_change_time))
            height, width = stdscr.getmaxyx()
            if draw_context is not None:
                style = str(draw_context.get("style", "classic"))
                if style in {"claude", "codex"} and _is_premium_layout(height, width, style):
                    input_width = _premium_input_width(width)
                elif style == "classic":
                    input_width = max(1, width - 6)
                else:
                    input_width = max(1, width - 4)
            else:
                input_width = max(1, width - 6)
            input_text, input_cursor = editor.get_display(input_width)
            status = f"{frame} {base_status} ({elapsed}s)"
            if progress:
                status = f"{status} {progress}"
            elif current_progress_count is not None:
                status = f"{status} calls:{current_progress_count}"
            if current_progress_count == 0:
                if request_timeout_seconds and request_timeout_seconds > 0:
                    status = f"{status} first-call<=~{int(request_timeout_seconds)}s"
                else:
                    status = f"{status} first-call-pending"
            if current_progress_count is not None:
                status = f"{status} idle:{idle_seconds}s"
                if idle_seconds >= stall_warn_seconds:
                    status = f"{status} possible-stall"
                    if current_progress_count == 0:
                        status = f"{status} (waiting on first backend call)"
            if draw_context is not None:
                _draw_main_screen(
                    stdscr,
                    transcript,
                    status,
                    mode_label,
                    input_text,
                    input_cursor,
                    scroll_offset,
                    density,
                    theme,
                    metrics_badge,
                    style=draw_context["style"],
                    model=draw_context["model"],
                    reasoning_effort=draw_context["reasoning_effort"],
                    cwd=draw_context["cwd"],
                    session_id=draw_context["session_id"],
                    autosave_path=draw_context["autosave_path"],
                    context_enabled=draw_context["context_enabled"],
                    placeholder=draw_context["placeholder"],
                    activity=draw_context.get("activity", []),
                    charset=draw_context.get("charset", "ascii"),
                )
            else:
                _draw_screen(
                    stdscr,
                    transcript,
                    status,
                    mode_label,
                    input_text,
                    input_cursor,
                    scroll_offset,
                    density,
                    theme,
                    metrics_badge,
                )
            ch = stdscr.getch()
            if ch == curses.KEY_RESIZE:
                stdscr.erase()
            elif ch in (3, ord("q")):
                if cancel_event is not None:
                    cancel_event.set()
            frame_idx += 1
    finally:
        stdscr.timeout(-1)

    thread.join()
    if "error" in failure:
        raise failure["error"]
    return outcome.get("value")


def _run_tui_session(
    args: argparse.Namespace,
    rlm: Any,
    lm: Any,
    input_field: str,
    output_field: str,
) -> int:
    turns: list[tuple[str, str]] = []
    session_id = _new_session_id("rlm")
    session_file = _sessions_dir() / f"{session_id}.json"
    session_state: dict[str, Any] = {
        "version": 1,
        "session_id": session_id,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "mode": "rlm",
        "cwd": args.cwd,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "theme": args.tui_theme,
        "density": args.tui_density,
        "style": args.tui_style,
        "charset": args.tui_charset,
        "no_history": bool(args.no_history),
        "activity": [],
        "turns": [],
    }

    def _persist_session() -> None:
        session_state["updated_at"] = _utc_now_iso()
        try:
            _atomic_write_text(session_file, json.dumps(session_state, ensure_ascii=False, indent=2))
        except Exception:
            # Best-effort persistence only; the TUI must remain usable without state writes.
            return

    _persist_session()

    transcript: list[tuple[str, str]] = []
    initial_status = "Ready."
    last_prediction: Any = None
    last_run_stats = "No run stats yet."
    metrics_badge = _metrics_badge_from_stats(last_run_stats, "rlm")
    last_user_message: str | None = None
    last_answer: str | None = None

    def _session(stdscr: Any) -> None:
        nonlocal last_prediction, last_run_stats, metrics_badge, session_id, session_file, session_state, last_user_message, last_answer
        scroll_offset = 0
        editor = _LineEditor(history_path=_history_path("rlm"))
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.keypad(True)
        _init_colors(args.tui_theme)
        tui_charset = _resolve_tui_charset(getattr(args, "tui_charset", "auto"))
        if not args.no_boot_animation:
            _boot_animation(stdscr, args.tui_theme, tui_charset)

        placeholder = 'Try \"summarize this repo\"'
        status = initial_status
        while True:
            height, width = stdscr.getmaxyx()
            if _is_premium_layout(height, width, args.tui_style):
                header_h = _premium_header_height(args.tui_style)  # type: ignore[arg-type]
                body_height = max(1, height - header_h - 4)
            else:
                body_height = _transcript_body_height(height, width)
            step = max(1, body_height // 2)
            if _is_premium_layout(height, width, args.tui_style):
                input_width = _premium_input_width(width)
            else:
                input_width = max(1, width - (6 if args.tui_style == "classic" else 4))
            input_text, input_cursor = editor.get_display(input_width)
            _draw_main_screen(
                stdscr,
                transcript,
                status,
                "rlm",
                input_text,
                input_cursor,
                scroll_offset,
                args.tui_density,
                args.tui_theme,
                metrics_badge,
                style=args.tui_style,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                cwd=args.cwd,
                session_id=session_id,
                autosave_path=str(session_file),
                context_enabled=not args.no_history,
                placeholder=placeholder,
                activity=session_state.get("activity", []) if isinstance(session_state.get("activity"), list) else [],
                charset=tui_charset,
            )
            ch = stdscr.getch()

            if ch == 27:  # ESC (used for bracketed paste sequences)
                token = _consume_bracketed_paste(stdscr, restore_timeout=-1)
                if token == "paste_start":
                    editor.start_paste()
                    status = "Paste started. (Newlines will not auto-send.)"
                    continue
                if token == "paste_end":
                    editor.end_paste()
                    status = "Paste complete. Press Enter to send, or /edit to review."
                    continue

            if ch == curses.KEY_RESIZE:
                scroll_offset = 0
                status = "Resized."
                continue
            if ch == curses.KEY_PPAGE:
                if _is_premium_layout(height, width, args.tui_style):
                    lines = _render_transcript_lines_premium(
                        transcript,
                        max(1, width - 2),
                        density=args.tui_density,
                    )
                else:
                    lines = _render_transcript_lines(
                        transcript,
                        max(1, width - 1),
                        density=args.tui_density,
                        theme=args.tui_theme,
                    )
                _, _, max_scroll = _slice_transcript_lines(lines, body_height, scroll_offset)
                scroll_offset = min(max_scroll, scroll_offset + step)
                continue
            if ch == curses.KEY_NPAGE:
                scroll_offset = max(0, scroll_offset - step)
                continue

            if ch == 16:  # Ctrl+P (command palette)
                cmd = _command_palette(stdscr, args.tui_theme, tui_charset)
                if cmd:
                    editor.set_text(cmd)
                    status = f"Inserted {cmd.strip()}"
                else:
                    status = initial_status
                continue

            if ch == 9 and _handle_tab_completion(stdscr, editor, args.tui_theme):
                status = initial_status
                continue

            if ch == ord("?") and not editor.get_text():
                _pager(
                    stdscr,
                    "Shortcuts",
                    [
                        "/help  show commands",
                        "/edit  open $EDITOR to compose",
                        "/trace [N|full]  view trajectory",
                        "/stats  show last run stats",
                        "/sessions  list recent sessions",
                        "/context on|off  toggle conversation context",
                        "/style claude|codex|classic",
                        "^P command palette",
                        "PgUp/PgDn  scroll",
                        "Tab  complete commands",
                        "Ctrl+C  cancel in-flight request",
                        "Ctrl+L  clear session",
                        "",
                        "Tip: type //help to send a literal /help.",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue

            if ch == 12:  # Ctrl+L
                turns.clear()
                transcript.clear()
                session_state["turns"] = []
                _persist_session()
                last_user_message = None
                last_answer = None
                editor.clear()
                scroll_offset = 0
                status = "Cleared."
                continue

            user_input = editor.feed(ch)
            if user_input is None:
                continue
            if user_input == "":
                status = "Cleared current input."
                continue

            # Escape a leading slash: `//help` becomes literal `/help`.
            if "\n" not in user_input and user_input.startswith("//"):
                user_input = user_input[1:]

            is_command = "\n" not in user_input and user_input.startswith("/")

            if is_command and user_input in {"/quit", "/exit"}:
                break

            if is_command:
                _record_activity(session_state, user_input)
                _persist_session()

            if is_command and user_input == "/help":
                _pager(
                    stdscr,
                    "Help",
                    [
                        "Core: /help /quit /clear /stats /trace [N|full]",
                        "Compose: /edit [last]  (opens $EDITOR)",
                        "Session: /sessions /load <id|path> /session /export md|json [path]",
                        "Runtime: /model [name] /reasoning [effort] /theme [name] /density [name] /style [name]",
                        "Display: /charset [auto|ascii|unicode]",
                        "Context: /context on|off  (toggle conversation context)",
                        "Copy: /copy [last|all]",
                        "",
                        "Keys: PgUp/PgDn scroll | Up/Down history | Tab complete | Ctrl+C cancel | Ctrl+L clear",
                        "Keys: Ctrl+P opens command palette",
                        "Tip: type //help to send a literal /help.",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue

            if is_command and user_input == "/stats":
                _pager(stdscr, "Stats", [last_run_stats], args.tui_theme)
                status = initial_status
                continue
            if is_command and user_input.startswith("/trace"):
                parts = user_input.split(maxsplit=1)
                arg = parts[1].strip() if len(parts) > 1 else ""
                if last_prediction is None:
                    _pager(stdscr, "Trace", ["No run stats yet."], args.tui_theme)
                else:
                    if not arg:
                        max_steps: int | None = 25
                    elif arg.lower() in {"full", "all"}:
                        max_steps = None
                    else:
                        try:
                            max_steps = max(1, int(arg))
                        except ValueError:
                            max_steps = 25
                    _pager(stdscr, "Trace", _trajectory_full_lines(last_prediction, max_steps=max_steps), args.tui_theme)
                status = initial_status
                continue

            if is_command and user_input == "/edit":
                edited = _edit_text_in_editor_from_curses(stdscr, "", args.tui_theme)
                editor.set_text(edited.rstrip("\n"))
                status = "Draft loaded from editor. Press Enter to send."
                continue
            if is_command and user_input == "/edit last":
                seed = last_user_message or ""
                edited = _edit_text_in_editor_from_curses(stdscr, seed, args.tui_theme)
                editor.set_text(edited.rstrip("\n"))
                status = "Draft loaded from editor. Press Enter to send."
                continue

            if is_command and user_input == "/retry":
                if last_user_message:
                    editor.set_text(last_user_message)
                    status = "Draft loaded. Press Enter to resend."
                else:
                    status = "Nothing to retry yet."
                continue

            if is_command and user_input.startswith("/context"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Context: {'off' if args.no_history else 'on'}"
                else:
                    val = parts[1].strip().lower()
                    if val in {"on", "yes", "true", "1"}:
                        args.no_history = False
                    elif val in {"off", "no", "false", "0"}:
                        args.no_history = True
                    else:
                        status = "Usage: /context [on|off]"
                        continue
                    session_state["no_history"] = bool(args.no_history)
                    _persist_session()
                    status = f"Context {'disabled' if args.no_history else 'enabled'}."
                continue

            if is_command and user_input.startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Model: {getattr(lm, 'model', args.model)}"
                else:
                    new_model = parts[1].strip()
                    if new_model:
                        args.model = new_model
                        try:
                            setattr(lm, "model", new_model)
                        except Exception:
                            pass
                        session_state["model"] = new_model
                        _persist_session()
                        status = f"Model set: {new_model}"
                continue

            if is_command and user_input.startswith("/reasoning"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Reasoning: {getattr(lm, 'reasoning_effort', args.reasoning_effort)}"
                else:
                    effort = parts[1].strip()
                    if effort:
                        args.reasoning_effort = effort
                        try:
                            setattr(lm, "reasoning_effort", effort)
                        except Exception:
                            pass
                        session_state["reasoning_effort"] = effort
                        _persist_session()
                        status = f"Reasoning effort set: {effort}"
                continue

            if is_command and user_input.startswith("/theme"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Theme: {args.tui_theme}"
                else:
                    theme = parts[1].strip()
                    if theme in _THEME_PALETTES:
                        args.tui_theme = theme
                        _init_colors(args.tui_theme)
                        session_state["theme"] = theme
                        _persist_session()
                        status = f"Theme set: {theme}"
                    else:
                        status = f"Unknown theme: {theme}"
                continue

            if is_command and user_input.startswith("/density"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Density: {args.tui_density}"
                else:
                    density = parts[1].strip()
                    if density in {"compact", "cozy"}:
                        args.tui_density = density
                        session_state["density"] = density
                        _persist_session()
                        status = f"Density set: {density}"
                    else:
                        status = f"Unknown density: {density}"
                continue

            if is_command and user_input.startswith("/style"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Style: {args.tui_style}"
                else:
                    style = parts[1].strip()
                    if style in {"claude", "codex", "classic"}:
                        args.tui_style = style
                        session_state["style"] = style
                        _persist_session()
                        status = f"Style set: {style}"
                    else:
                        status = f"Unknown style: {style}"
                continue

            if is_command and user_input.startswith("/charset"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Charset: {args.tui_charset}"
                else:
                    val = parts[1].strip().lower()
                    if val in {"auto", "ascii", "unicode"}:
                        args.tui_charset = val
                        tui_charset = _resolve_tui_charset(val)
                        session_state["charset"] = val
                        _persist_session()
                        status = f"Charset set: {val}"
                    else:
                        status = "Usage: /charset [auto|ascii|unicode]"
                continue

            if is_command and user_input.startswith("/copy"):
                parts = user_input.split(maxsplit=1)
                target = parts[1].strip().lower() if len(parts) > 1 else "last"
                if target in {"last", ""}:
                    ok, provider = _copy_to_clipboard(last_answer or "")
                    status = "Copied." if ok else f"Copy failed ({provider})."
                else:
                    ok, provider = _copy_to_clipboard("\n".join(f"[{r}] {t}" for r, t in transcript))
                    status = "Copied transcript." if ok else f"Copy failed ({provider})."
                continue

            if is_command and user_input.startswith("/export"):
                try:
                    parts = shlex.split(user_input)
                except ValueError:
                    status = "Usage: /export (md|json) [path]"
                    continue
                fmt = parts[1] if len(parts) > 1 else "md"
                out_path = Path(parts[2]).expanduser() if len(parts) > 2 else None
                if fmt not in {"md", "json"}:
                    status = "Usage: /export (md|json) [path]"
                    continue
                if out_path is None:
                    out_path = _sessions_dir() / f"{session_id}.{fmt}"
                if fmt == "json":
                    try:
                        _atomic_write_text(out_path, json.dumps(session_state, ensure_ascii=False, indent=2))
                    except Exception as e:
                        status = f"Export failed: {e}"
                        continue
                else:
                    md_lines: list[str] = []
                    md_lines.append(f"# RLM session {session_id}")
                    md_lines.append("")
                    md_lines.append(f"- mode: rlm")
                    md_lines.append(f"- model: {session_state.get('model')}")
                    md_lines.append(f"- reasoning_effort: {session_state.get('reasoning_effort')}")
                    md_lines.append(f"- cwd: {session_state.get('cwd')}")
                    md_lines.append("")
                    for t in session_state.get("turns", []):
                        if not isinstance(t, dict):
                            continue
                        md_lines.append("## You")
                        md_lines.append(str(t.get("user", "")))
                        md_lines.append("")
                        md_lines.append("## RLM")
                        md_lines.append(str(t.get("assistant", "")))
                        md_lines.append("")
                    try:
                        _atomic_write_text(out_path, "\n".join(md_lines) + "\n")
                    except Exception as e:
                        status = f"Export failed: {e}"
                        continue
                status = f"Exported: {out_path}"
                continue

            if is_command and user_input == "/sessions":
                items: list[str] = []
                for p in sorted(_sessions_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                    items.append(p.name)
                if not items:
                    _pager(stdscr, "Sessions", ["No sessions found."], args.tui_theme)
                else:
                    _pager(
                        stdscr,
                        "Sessions",
                        ["Recent sessions (newest first):", "", *items, "", "Tip: /load <filename> to open."],
                        args.tui_theme,
                    )
                status = initial_status
                continue

            if is_command and user_input.startswith("/load "):
                token = user_input.split(maxsplit=1)[1].strip()
                candidate = Path(token).expanduser()
                if not candidate.exists():
                    direct = _sessions_dir() / token
                    if direct.suffix != ".json":
                        direct = direct.with_suffix(".json")
                    if direct.exists():
                        candidate = direct
                    else:
                        matches = sorted(_sessions_dir().glob(f"{token}*.json"))
                        if matches:
                            candidate = matches[0]
                if not candidate.exists():
                    status = f"Session not found: {token}"
                    continue
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    status = f"Invalid session file: {candidate}"
                    continue
                if not isinstance(data, dict):
                    status = f"Invalid session file: {candidate}"
                    continue
                session_id = str(data.get("session_id", session_id))
                session_file = candidate
                session_state = data
                if not isinstance(session_state.get("activity"), list):
                    session_state["activity"] = []
                args.no_history = bool(session_state.get("no_history", args.no_history))
                loaded_theme = session_state.get("theme")
                if isinstance(loaded_theme, str) and loaded_theme in _THEME_PALETTES:
                    args.tui_theme = loaded_theme
                    _init_colors(args.tui_theme)
                loaded_density = session_state.get("density")
                if isinstance(loaded_density, str) and loaded_density in {"compact", "cozy"}:
                    args.tui_density = loaded_density
                loaded_style = session_state.get("style")
                if isinstance(loaded_style, str) and loaded_style in {"claude", "codex", "classic"}:
                    args.tui_style = loaded_style
                loaded_charset = session_state.get("charset")
                if isinstance(loaded_charset, str) and loaded_charset in {"auto", "ascii", "unicode"}:
                    args.tui_charset = loaded_charset
                    tui_charset = _resolve_tui_charset(loaded_charset)
                loaded_model = session_state.get("model")
                if isinstance(loaded_model, str) and loaded_model:
                    args.model = loaded_model
                    try:
                        setattr(lm, "model", loaded_model)
                    except Exception:
                        pass
                loaded_effort = session_state.get("reasoning_effort")
                if isinstance(loaded_effort, str) and loaded_effort:
                    args.reasoning_effort = loaded_effort
                    try:
                        setattr(lm, "reasoning_effort", loaded_effort)
                    except Exception:
                        pass
                turns.clear()
                transcript.clear()
                for t in session_state.get("turns", []):
                    if not isinstance(t, dict):
                        continue
                    u = str(t.get("user", ""))
                    a = str(t.get("assistant", ""))
                    turns.append((u, a))
                    transcript.append(("you", u))
                    transcript.append(("rlm", a))
                if turns:
                    last_user_message, last_answer = turns[-1]
                else:
                    last_user_message, last_answer = None, None
                status = f"Loaded: {candidate}"
                continue

            if is_command and user_input == "/session":
                _pager(
                    stdscr,
                    "Session",
                    [
                        f"session_id: {session_id}",
                        f"autosave:   {session_file}",
                        f"model:      {args.model}",
                        f"reasoning:  {args.reasoning_effort}",
                        f"dir:        {args.cwd}",
                        f"context:    {'off' if args.no_history else 'on'}",
                        f"theme:      {args.tui_theme}",
                        f"density:    {args.tui_density}",
                        f"style:      {args.tui_style}",
                        f"charset:    {args.tui_charset}",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue

            if is_command and user_input == "/clear":
                turns.clear()
                transcript.clear()
                session_state["turns"] = []
                _persist_session()
                last_user_message = None
                last_answer = None
                editor.clear()
                scroll_offset = 0
                status = "Cleared."
                continue

            # Normal message.
            last_user_message = user_input
            _record_activity(session_state, f"ask: {_preview_inline(user_input, max_chars=42)}")
            _persist_session()
            transcript.append(("you", user_input))
            scroll_offset = 0
            status = "Running RLM..."

            try:
                start_history = _history_count(lm)
                cancel_event = threading.Event()

                def _invoke() -> Any:
                    setattr(lm, "cancel_event", cancel_event)
                    try:
                        query = user_input if args.no_history else _build_chat_query(turns, user_input)
                        return rlm(**{input_field: query})
                    finally:
                        try:
                            setattr(lm, "cancel_event", None)
                        except Exception:
                            pass

                def _progress() -> str:
                    return f"calls:{max(0, _history_count(lm) - start_history)}"

                def _progress_count() -> int:
                    return max(0, _history_count(lm) - start_history)

                prediction = _run_with_spinner(
                    stdscr,
                    transcript,
                    "rlm",
                    "Running RLM...",
                    _invoke,
                    editor,
                    scroll_offset,
                    args.tui_density,
                    args.tui_theme,
                    metrics_badge,
                    progress_fn=_progress,
                    progress_count_fn=_progress_count,
                    request_timeout_seconds=args.timeout_seconds,
                    cancel_event=cancel_event,
                    draw_context={
                        "style": args.tui_style,
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort,
                        "cwd": args.cwd,
                        "session_id": session_id,
                        "autosave_path": str(session_file),
                        "context_enabled": not args.no_history,
                        "placeholder": placeholder,
                        "activity": session_state.get("activity", []) if isinstance(session_state.get("activity"), list) else [],
                        "charset": tui_charset,
                    },
                )
                last_prediction = prediction
                lm_calls_delta = max(0, _history_count(lm) - start_history)
                last_run_stats = _build_rlm_run_stats(prediction, lm_calls_delta=lm_calls_delta, max_depth=args.max_depth)
                metrics_badge = _metrics_badge_from_stats(last_run_stats, "rlm")
                outputs = {name: getattr(prediction, name) for name in rlm.signature.output_fields}
                answer = str(outputs[output_field])
                turns.append((user_input, answer))
                last_answer = answer
                transcript.append(("rlm", answer))
                usage = lm.history[-1]["usage"] if lm.history else {}
                status = _usage_summary(usage)
                _record_activity(session_state, f"reply: {_preview_inline(answer, max_chars=42)}")

                session_state.setdefault("turns", []).append({
                    "ts": _utc_now_iso(),
                    "user": user_input,
                    "assistant": answer,
                    "run_stats": last_run_stats,
                    "usage": usage,
                })
                _persist_session()
            except Exception as e:
                transcript.append(("error", str(e)))
                if isinstance(e, TimeoutError):
                    status = "Timed out."
                elif e.__class__.__name__ == "CodexCLICancelled":
                    status = "Cancelled."
                else:
                    status = "Last call failed."

    _set_bracketed_paste(True)
    try:
        curses.wrapper(_session)
    finally:
        _set_bracketed_paste(False)
    return 0


def _run_chat_tui_session(args: argparse.Namespace, lm: Any) -> int:
    turns: list[tuple[str, str]] = []
    session_id = _new_session_id("chat")
    session_file = _sessions_dir() / f"{session_id}.json"
    session_state: dict[str, Any] = {
        "version": 1,
        "session_id": session_id,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "mode": "chat",
        "cwd": args.cwd,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "theme": args.tui_theme,
        "density": args.tui_density,
        "style": args.tui_style,
        "charset": args.tui_charset,
        "no_history": bool(args.no_history),
        "activity": [],
        "turns": [],
    }

    def _persist_session() -> None:
        session_state["updated_at"] = _utc_now_iso()
        try:
            _atomic_write_text(session_file, json.dumps(session_state, ensure_ascii=False, indent=2))
        except Exception:
            return

    _persist_session()

    transcript: list[tuple[str, str]] = []
    initial_status = "Ready."
    last_run_stats = "No run stats yet."
    metrics_badge = _metrics_badge_from_stats(last_run_stats, "chat")
    last_user_message: str | None = None
    last_answer: str | None = None

    def _session(stdscr: Any) -> None:
        nonlocal last_run_stats, metrics_badge, session_id, session_file, session_state, last_user_message, last_answer
        scroll_offset = 0
        editor = _LineEditor(history_path=_history_path("chat"))
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.keypad(True)
        _init_colors(args.tui_theme)
        tui_charset = _resolve_tui_charset(getattr(args, "tui_charset", "auto"))
        if not args.no_boot_animation:
            _boot_animation(stdscr, args.tui_theme, tui_charset)

        placeholder = 'Try \"explain this codebase\"'
        status = initial_status
        while True:
            height, width = stdscr.getmaxyx()
            if _is_premium_layout(height, width, args.tui_style):
                header_h = _premium_header_height(args.tui_style)  # type: ignore[arg-type]
                body_height = max(1, height - header_h - 4)
            else:
                body_height = _transcript_body_height(height, width)
            step = max(1, body_height // 2)
            if _is_premium_layout(height, width, args.tui_style):
                input_width = _premium_input_width(width)
            else:
                input_width = max(1, width - (6 if args.tui_style == "classic" else 4))
            input_text, input_cursor = editor.get_display(input_width)
            _draw_main_screen(
                stdscr,
                transcript,
                status,
                "chat",
                input_text,
                input_cursor,
                scroll_offset,
                args.tui_density,
                args.tui_theme,
                metrics_badge,
                style=args.tui_style,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                cwd=args.cwd,
                session_id=session_id,
                autosave_path=str(session_file),
                context_enabled=not args.no_history,
                placeholder=placeholder,
                activity=session_state.get("activity", []) if isinstance(session_state.get("activity"), list) else [],
                charset=tui_charset,
            )
            ch = stdscr.getch()

            if ch == 27:  # ESC (used for bracketed paste sequences)
                token = _consume_bracketed_paste(stdscr, restore_timeout=-1)
                if token == "paste_start":
                    editor.start_paste()
                    status = "Paste started. (Newlines will not auto-send.)"
                    continue
                if token == "paste_end":
                    editor.end_paste()
                    status = "Paste complete. Press Enter to send, or /edit to review."
                    continue

            if ch == curses.KEY_RESIZE:
                scroll_offset = 0
                status = "Resized."
                continue
            if ch == curses.KEY_PPAGE:
                if _is_premium_layout(height, width, args.tui_style):
                    lines = _render_transcript_lines_premium(
                        transcript,
                        max(1, width - 2),
                        density=args.tui_density,
                    )
                else:
                    lines = _render_transcript_lines(
                        transcript,
                        max(1, width - 1),
                        density=args.tui_density,
                        theme=args.tui_theme,
                    )
                _, _, max_scroll = _slice_transcript_lines(lines, body_height, scroll_offset)
                scroll_offset = min(max_scroll, scroll_offset + step)
                continue
            if ch == curses.KEY_NPAGE:
                scroll_offset = max(0, scroll_offset - step)
                continue

            if ch == 16:  # Ctrl+P (command palette)
                cmd = _command_palette(stdscr, args.tui_theme, tui_charset)
                if cmd:
                    editor.set_text(cmd)
                    status = f"Inserted {cmd.strip()}"
                else:
                    status = initial_status
                continue

            if ch == 9 and _handle_tab_completion(stdscr, editor, args.tui_theme):
                status = initial_status
                continue

            if ch == ord("?") and not editor.get_text():
                _pager(
                    stdscr,
                    "Shortcuts",
                    [
                        "/help  show commands",
                        "/edit  open $EDITOR to compose",
                        "/stats  show last run stats",
                        "/sessions  list recent sessions",
                        "/context on|off  toggle conversation context",
                        "/style claude|codex|classic",
                        "^P command palette",
                        "PgUp/PgDn  scroll",
                        "Tab  complete commands",
                        "Ctrl+C  cancel in-flight request",
                        "Ctrl+L  clear session",
                        "",
                        "Tip: type //help to send a literal /help.",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue

            if ch == 12:  # Ctrl+L
                turns.clear()
                transcript.clear()
                session_state["turns"] = []
                _persist_session()
                last_user_message = None
                last_answer = None
                editor.clear()
                scroll_offset = 0
                status = "Cleared."
                continue

            user_input = editor.feed(ch)
            if user_input is None:
                continue
            if user_input == "":
                status = "Cleared current input."
                continue

            if "\n" not in user_input and user_input.startswith("//"):
                user_input = user_input[1:]

            is_command = "\n" not in user_input and user_input.startswith("/")

            if is_command and user_input in {"/quit", "/exit"}:
                break
            if is_command and user_input == "/help":
                _pager(
                    stdscr,
                    "Help",
                    [
                        "Core: /help /quit /clear /stats",
                        "Compose: /edit [last]  (opens $EDITOR)",
                        "Session: /sessions /load <id|path> /session /export md|json [path]",
                        "Runtime: /model [name] /reasoning [effort] /theme [name] /density [name] /style [name]",
                        "Display: /charset [auto|ascii|unicode]",
                        "Context: /context on|off  (toggle conversation context)",
                        "Copy: /copy [last|all]",
                        "",
                        "Keys: PgUp/PgDn scroll | Up/Down history | Tab complete | Ctrl+C cancel | Ctrl+L clear",
                        "Keys: Ctrl+P opens command palette",
                        "Tip: type //help to send a literal /help.",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue
            if is_command and user_input == "/stats":
                _pager(stdscr, "Stats", [last_run_stats], args.tui_theme)
                status = initial_status
                continue
            if is_command and user_input.startswith("/trace"):
                _pager(stdscr, "Trace", ["Trajectory is unavailable in chat mode. Use --mode rlm."], args.tui_theme)
                status = initial_status
                continue
            if is_command and user_input == "/edit":
                edited = _edit_text_in_editor_from_curses(stdscr, "", args.tui_theme)
                editor.set_text(edited.rstrip("\n"))
                status = "Draft loaded from editor. Press Enter to send."
                continue
            if is_command and user_input == "/edit last":
                seed = last_user_message or ""
                edited = _edit_text_in_editor_from_curses(stdscr, seed, args.tui_theme)
                editor.set_text(edited.rstrip("\n"))
                status = "Draft loaded from editor. Press Enter to send."
                continue

            if is_command and user_input == "/retry":
                if last_user_message:
                    editor.set_text(last_user_message)
                    status = "Draft loaded. Press Enter to resend."
                else:
                    status = "Nothing to retry yet."
                continue

            if is_command and user_input.startswith("/context"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Context: {'off' if args.no_history else 'on'}"
                else:
                    val = parts[1].strip().lower()
                    if val in {"on", "yes", "true", "1"}:
                        args.no_history = False
                    elif val in {"off", "no", "false", "0"}:
                        args.no_history = True
                    else:
                        status = "Usage: /context [on|off]"
                        continue
                    session_state["no_history"] = bool(args.no_history)
                    _persist_session()
                    status = f"Context {'disabled' if args.no_history else 'enabled'}."
                continue

            if is_command and user_input.startswith("/model"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Model: {getattr(lm, 'model', args.model)}"
                else:
                    new_model = parts[1].strip()
                    if new_model:
                        args.model = new_model
                        try:
                            setattr(lm, "model", new_model)
                        except Exception:
                            pass
                        session_state["model"] = new_model
                        _persist_session()
                        status = f"Model set: {new_model}"
                continue

            if is_command and user_input.startswith("/reasoning"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Reasoning: {getattr(lm, 'reasoning_effort', args.reasoning_effort)}"
                else:
                    effort = parts[1].strip()
                    if effort:
                        args.reasoning_effort = effort
                        try:
                            setattr(lm, "reasoning_effort", effort)
                        except Exception:
                            pass
                        session_state["reasoning_effort"] = effort
                        _persist_session()
                        status = f"Reasoning effort set: {effort}"
                continue

            if is_command and user_input.startswith("/theme"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Theme: {args.tui_theme}"
                else:
                    theme = parts[1].strip()
                    if theme in _THEME_PALETTES:
                        args.tui_theme = theme
                        _init_colors(args.tui_theme)
                        session_state["theme"] = theme
                        _persist_session()
                        status = f"Theme set: {theme}"
                    else:
                        status = f"Unknown theme: {theme}"
                continue

            if is_command and user_input.startswith("/density"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Density: {args.tui_density}"
                else:
                    density = parts[1].strip()
                    if density in {"compact", "cozy"}:
                        args.tui_density = density
                        session_state["density"] = density
                        _persist_session()
                        status = f"Density set: {density}"
                    else:
                        status = f"Unknown density: {density}"
                continue

            if is_command and user_input.startswith("/style"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Style: {args.tui_style}"
                else:
                    style = parts[1].strip()
                    if style in {"claude", "codex", "classic"}:
                        args.tui_style = style
                        session_state["style"] = style
                        _persist_session()
                        status = f"Style set: {style}"
                    else:
                        status = f"Unknown style: {style}"
                continue

            if is_command and user_input.startswith("/charset"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    status = f"Charset: {args.tui_charset}"
                else:
                    val = parts[1].strip().lower()
                    if val in {"auto", "ascii", "unicode"}:
                        args.tui_charset = val
                        tui_charset = _resolve_tui_charset(val)
                        session_state["charset"] = val
                        _persist_session()
                        status = f"Charset set: {val}"
                    else:
                        status = "Usage: /charset [auto|ascii|unicode]"
                continue

            if is_command and user_input.startswith("/copy"):
                parts = user_input.split(maxsplit=1)
                target = parts[1].strip().lower() if len(parts) > 1 else "last"
                if target in {"last", ""}:
                    ok, provider = _copy_to_clipboard(last_answer or "")
                    status = "Copied." if ok else f"Copy failed ({provider})."
                else:
                    ok, provider = _copy_to_clipboard("\n".join(f"[{r}] {t}" for r, t in transcript))
                    status = "Copied transcript." if ok else f"Copy failed ({provider})."
                continue

            if is_command and user_input.startswith("/export"):
                try:
                    parts = shlex.split(user_input)
                except ValueError:
                    status = "Usage: /export (md|json) [path]"
                    continue
                fmt = parts[1] if len(parts) > 1 else "md"
                out_path = Path(parts[2]).expanduser() if len(parts) > 2 else None
                if fmt not in {"md", "json"}:
                    status = "Usage: /export (md|json) [path]"
                    continue
                if out_path is None:
                    out_path = _sessions_dir() / f"{session_id}.{fmt}"
                if fmt == "json":
                    try:
                        _atomic_write_text(out_path, json.dumps(session_state, ensure_ascii=False, indent=2))
                    except Exception as e:
                        status = f"Export failed: {e}"
                        continue
                else:
                    md_lines: list[str] = []
                    md_lines.append(f"# Chat session {session_id}")
                    md_lines.append("")
                    md_lines.append(f"- mode: chat")
                    md_lines.append(f"- model: {session_state.get('model')}")
                    md_lines.append(f"- reasoning_effort: {session_state.get('reasoning_effort')}")
                    md_lines.append(f"- cwd: {session_state.get('cwd')}")
                    md_lines.append("")
                    for t in session_state.get("turns", []):
                        if not isinstance(t, dict):
                            continue
                        md_lines.append("## You")
                        md_lines.append(str(t.get("user", "")))
                        md_lines.append("")
                        md_lines.append("## Assistant")
                        md_lines.append(str(t.get("assistant", "")))
                        md_lines.append("")
                    try:
                        _atomic_write_text(out_path, "\n".join(md_lines) + "\n")
                    except Exception as e:
                        status = f"Export failed: {e}"
                        continue
                status = f"Exported: {out_path}"
                continue

            if is_command and user_input == "/sessions":
                items: list[str] = []
                for p in sorted(_sessions_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                    items.append(p.name)
                if not items:
                    _pager(stdscr, "Sessions", ["No sessions found."], args.tui_theme)
                else:
                    _pager(
                        stdscr,
                        "Sessions",
                        ["Recent sessions (newest first):", "", *items, "", "Tip: /load <filename> to open."],
                        args.tui_theme,
                    )
                status = initial_status
                continue

            if is_command and user_input.startswith("/load "):
                token = user_input.split(maxsplit=1)[1].strip()
                candidate = Path(token).expanduser()
                if not candidate.exists():
                    direct = _sessions_dir() / token
                    if direct.suffix != ".json":
                        direct = direct.with_suffix(".json")
                    if direct.exists():
                        candidate = direct
                    else:
                        matches = sorted(_sessions_dir().glob(f"{token}*.json"))
                        if matches:
                            candidate = matches[0]
                if not candidate.exists():
                    status = f"Session not found: {token}"
                    continue
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    status = f"Invalid session file: {candidate}"
                    continue
                if not isinstance(data, dict):
                    status = f"Invalid session file: {candidate}"
                    continue
                session_id = str(data.get("session_id", session_id))
                session_file = candidate
                session_state = data
                if not isinstance(session_state.get("activity"), list):
                    session_state["activity"] = []
                args.no_history = bool(session_state.get("no_history", args.no_history))
                loaded_theme = session_state.get("theme")
                if isinstance(loaded_theme, str) and loaded_theme in _THEME_PALETTES:
                    args.tui_theme = loaded_theme
                    _init_colors(args.tui_theme)
                loaded_density = session_state.get("density")
                if isinstance(loaded_density, str) and loaded_density in {"compact", "cozy"}:
                    args.tui_density = loaded_density
                loaded_style = session_state.get("style")
                if isinstance(loaded_style, str) and loaded_style in {"claude", "codex", "classic"}:
                    args.tui_style = loaded_style
                loaded_charset = session_state.get("charset")
                if isinstance(loaded_charset, str) and loaded_charset in {"auto", "ascii", "unicode"}:
                    args.tui_charset = loaded_charset
                    tui_charset = _resolve_tui_charset(loaded_charset)
                loaded_model = session_state.get("model")
                if isinstance(loaded_model, str) and loaded_model:
                    args.model = loaded_model
                    try:
                        setattr(lm, "model", loaded_model)
                    except Exception:
                        pass
                loaded_effort = session_state.get("reasoning_effort")
                if isinstance(loaded_effort, str) and loaded_effort:
                    args.reasoning_effort = loaded_effort
                    try:
                        setattr(lm, "reasoning_effort", loaded_effort)
                    except Exception:
                        pass
                turns.clear()
                transcript.clear()
                for t in session_state.get("turns", []):
                    if not isinstance(t, dict):
                        continue
                    u = str(t.get("user", ""))
                    a = str(t.get("assistant", ""))
                    turns.append((u, a))
                    transcript.append(("you", u))
                    transcript.append(("codex", a))
                if turns:
                    last_user_message, last_answer = turns[-1]
                else:
                    last_user_message, last_answer = None, None
                status = f"Loaded: {candidate}"
                continue

            if is_command and user_input == "/session":
                _pager(
                    stdscr,
                    "Session",
                    [
                        f"session_id: {session_id}",
                        f"autosave:   {session_file}",
                        f"model:      {args.model}",
                        f"reasoning:  {args.reasoning_effort}",
                        f"dir:        {args.cwd}",
                        f"context:    {'off' if args.no_history else 'on'}",
                        f"theme:      {args.tui_theme}",
                        f"density:    {args.tui_density}",
                        f"style:      {args.tui_style}",
                        f"charset:    {args.tui_charset}",
                    ],
                    args.tui_theme,
                )
                status = initial_status
                continue

            if is_command and user_input == "/clear":
                turns.clear()
                transcript.clear()
                session_state["turns"] = []
                _persist_session()
                last_user_message = None
                last_answer = None
                editor.clear()
                scroll_offset = 0
                status = "Cleared."
                continue

            last_user_message = user_input
            _record_activity(session_state, f"ask: {_preview_inline(user_input, max_chars=42)}")
            _persist_session()
            transcript.append(("you", user_input))
            scroll_offset = 0
            status = "Querying Codex..."

            try:
                start_history = _history_count(lm)
                cancel_event = threading.Event()

                def _invoke() -> list[Any]:
                    setattr(lm, "cancel_event", cancel_event)
                    try:
                        if args.no_history:
                            return lm(prompt=user_input)
                        return lm(messages=_build_chat_messages(turns, user_input))
                    finally:
                        try:
                            setattr(lm, "cancel_event", None)
                        except Exception:
                            pass

                def _progress() -> str:
                    return f"calls:{max(0, _history_count(lm) - start_history)}"

                def _progress_count() -> int:
                    return max(0, _history_count(lm) - start_history)

                outputs = _run_with_spinner(
                    stdscr,
                    transcript,
                    "chat",
                    "Querying Codex...",
                    _invoke,
                    editor,
                    scroll_offset,
                    args.tui_density,
                    args.tui_theme,
                    metrics_badge,
                    progress_fn=_progress,
                    progress_count_fn=_progress_count,
                    request_timeout_seconds=args.timeout_seconds,
                    cancel_event=cancel_event,
                    draw_context={
                        "style": args.tui_style,
                        "model": args.model,
                        "reasoning_effort": args.reasoning_effort,
                        "cwd": args.cwd,
                        "session_id": session_id,
                        "autosave_path": str(session_file),
                        "context_enabled": not args.no_history,
                        "placeholder": placeholder,
                        "activity": session_state.get("activity", []) if isinstance(session_state.get("activity"), list) else [],
                        "charset": tui_charset,
                    },
                )

                answer = _extract_text_output(outputs)
                lm_calls_delta = max(0, _history_count(lm) - start_history)
                last_run_stats = _build_chat_run_stats(lm_calls_delta=lm_calls_delta)
                metrics_badge = _metrics_badge_from_stats(last_run_stats, "chat")
                turns.append((user_input, answer))
                last_answer = answer
                transcript.append(("codex", answer))
                usage = lm.history[-1]["usage"] if lm.history else {}
                status = _usage_summary(usage)
                _record_activity(session_state, f"reply: {_preview_inline(answer, max_chars=42)}")

                session_state.setdefault("turns", []).append({
                    "ts": _utc_now_iso(),
                    "user": user_input,
                    "assistant": answer,
                    "run_stats": last_run_stats,
                    "usage": usage,
                })
                _persist_session()
            except Exception as e:
                transcript.append(("error", str(e)))
                if isinstance(e, TimeoutError):
                    status = "Timed out."
                elif e.__class__.__name__ == "CodexCLICancelled":
                    status = "Cancelled."
                else:
                    status = "Last call failed."

    _set_bracketed_paste(True)
    try:
        curses.wrapper(_session)
    finally:
        _set_bracketed_paste(False)
    return 0


def _build_runtime(args: argparse.Namespace, mode: Literal["rlm", "chat"]):
    signature = ensure_signature(args.signature)
    input_fields = list(signature.input_fields.keys())

    extra_config: dict[str, Any] = {}
    # Disable Codex shell tool by default. RLM has explicit, workspace-scoped file tools for context
    # management; letting Codex run arbitrary shell commands is an unnecessary footgun.
    extra_config["features.shell_tool"] = bool(args.codex_enable_shell_tool)
    if args.codex_trust_level:
        # Most-specific match wins. This keeps non-interactive codex exec runs from freely executing
        # untrusted commands in a "trusted" workspace.
        extra_config[f'projects."{args.cwd}".trust_level'] = args.codex_trust_level

    approval = args.codex_approval
    if approval is None:
        approval = "untrusted" if args.codex_enable_shell_tool else "never"

    lm = dspy.CodexCLI(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        command=args.codex_command,
        working_dir=args.cwd,
        prompt_mode=args.codex_prompt_mode if mode == "rlm" else "chat",
        sandbox=args.codex_sandbox,
        ask_for_approval=approval,
        codex_cd=args.cwd,
        extra_config=extra_config,
    )
    dspy.configure(lm=lm)

    if mode == "chat":
        return signature, input_fields, "answer", lm, None

    # NOTE: keep the tool name `workspace_root()` (used by the agent) but avoid
    # shadowing the `workspace_root` path value, which would break path
    # resolution for all file tools.
    workspace_root_path = os.path.realpath(args.cwd)

    def _resolve_workspace_path(path: str) -> str:
        if not path:
            raise ValueError("path cannot be empty")
        candidate = path
        if not os.path.isabs(candidate):
            candidate = os.path.join(workspace_root_path, candidate)
        real = os.path.realpath(candidate)
        if real != workspace_root_path and not real.startswith(workspace_root_path + os.sep):
            raise ValueError(f"Path is outside workspace_root: {path!r}")
        return real

    def workspace_root() -> str:
        """Return the absolute workspace root path (the only allowed root for file tools)."""
        return workspace_root_path

    def list_dir(path: str = ".", max_entries: int = 200) -> list[str]:
        """List directory entries under the workspace root. Returns a sorted list of names."""
        real = _resolve_workspace_path(path)
        if not os.path.isdir(real):
            raise ValueError(f"Not a directory: {path!r}")
        entries = sorted(os.listdir(real))
        if len(entries) > max_entries:
            return entries[:max_entries] + [f"... ({len(entries) - max_entries} more)"]
        return entries

    def read_text_file(path: str, max_bytes: int = 200_000) -> str:
        """Read a UTF-8 text file under the workspace root (truncated to max_bytes)."""
        real = _resolve_workspace_path(path)
        if os.path.isdir(real):
            raise ValueError(f"Path is a directory: {path!r}")
        with open(real, "rb") as f:
            data = f.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        if truncated:
            text += "\n\n[truncated]"
        return text

    rlm_tools: list[Callable] = [workspace_root, list_dir, read_text_file]
    interpreter = LocalInterpreter() if args.unsafe_local_interpreter else None

    rlm = dspy.RLM(
        args.signature,
        interpreter=interpreter,
        unsafe_local_subcalls=args.unsafe_local_subcalls,
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        max_cost=args.max_cost,
        verbose=args.verbose,
        tools=rlm_tools,
    )
    output_field = _pick_output_field(args, list(rlm.signature.output_fields.keys()))
    return signature, input_fields, output_field, lm, rlm


def _run_one_shot(args: argparse.Namespace, input_fields: list[str], output_field: str, lm: Any, rlm: Any) -> int:
    inputs = _read_inputs(args, input_fields)
    prediction = rlm(**inputs)
    outputs = {name: getattr(prediction, name) for name in rlm.signature.output_fields}
    usage = lm.history[-1]["usage"] if lm.history else {}

    if args.json:
        payload = {
            "inputs": inputs,
            "outputs": outputs,
            "usage": usage,
            "trajectory_length": len(getattr(prediction, "trajectory", [])),
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(outputs[output_field])
    if not args.quiet and usage:
        print(f"\nusage: {_usage_summary(usage)}")
    return 0


def _run_chat_one_shot(args: argparse.Namespace, lm: Any) -> int:
    inputs = _read_inputs(args, ["query"])
    if "query" in inputs:
        prompt = str(inputs["query"])
    elif len(inputs) == 1:
        prompt = str(next(iter(inputs.values())))
    else:
        raise ValueError("Chat mode expects a single input value or a JSON object with a 'query' field.")

    outputs = lm(prompt=prompt)
    answer = _extract_text_output(outputs)
    usage = lm.history[-1]["usage"] if lm.history else {}

    if args.json:
        payload = {
            "input": prompt,
            "output": answer,
            "usage": usage,
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    print(answer)
    if not args.quiet and usage:
        print(f"\nusage: {_usage_summary(usage)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        launch_tui = _should_launch_tui(args)
        mode = _resolve_mode(args, launch_tui)
        _, input_fields, output_field, lm, rlm = _build_runtime(args, mode)

        if launch_tui:
            if mode == "chat":
                return _run_chat_tui_session(args, lm)
            if len(input_fields) != 1:
                raise ValueError("TUI mode currently requires a single-input signature.")
            return _run_tui_session(args, rlm, lm, input_fields[0], output_field)

        if mode == "chat":
            return _run_chat_one_shot(args, lm)

        return _run_one_shot(args, input_fields, output_field, lm, rlm)
    except Exception as e:
        print(f"rlm error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
