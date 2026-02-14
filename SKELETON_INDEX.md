# Skeleton Index (DSPy RLM)

This is an at-a-glance map of the repo: the entrypoints, the major subsystems, the public symbols people actually use, and the "call graph" between them. It is intentionally not a file-tree dump.

## 1) Entrypoints

- **Python package**: `dspy/`
  - The library is imported via `import dspy` (public API is largely re-exported from `dspy/__init__.py`).
- **CLI**: `rlm`
  - Defined in `pyproject.toml` under `[project.scripts]`: `rlm = "dspy.cli.rlm:main"`.
  - Implementation: `dspy/cli/rlm.py` (one-shot mode + interactive TUI mode).
- **Docs site**: `docs/`
  - MkDocs config: `docs/mkdocs.yml`
  - Vercel config: `docs/vercel.json`
  - Content root: `docs/docs/` (e.g. `docs/docs/index.md`)
  - API docs generators: `docs/scripts/generate_api_docs.py`, `docs/scripts/generate_api_summary.py`
- **CI** (GitHub Actions):
  - Tests/lint/build: `.github/workflows/run_tests.yml`
  - Pre-commit checks (manual): `.github/workflows/precommits_check.yml`
  - Release/publish: `.github/workflows/build_and_release.yml`
    - Version helper: `.github/workflows/build_utils/test_version.py`
    - Internal `dspy-ai` manifest: `.github/.internal_dspyai/pyproject.toml`
    - Manual TestPyPI install helper: `.github/workflow_scripts/install_testpypi_pkg.sh`
  - Docs build/push: `.github/workflows/docs-push.yml`

## 2) Key Subsystems (Owning Directories)

- **Core runtime configuration**: `dspy/dsp/utils/settings.py`
  - Global `dspy.settings` / `dspy.configure(...)`, plus thread-local overrides via `dspy.context(...)`.
- **CLI / TUI**: `dspy/cli/`
  - `rlm` entrypoint (one-shot + interactive curses TUI); wires `CodexCLI` + `RLM` and safety flags.
- **LLM + embedding clients, caching**: `dspy/clients/`
  - `LM` (LiteLLM-backed), `Embedder`, `CodexCLI`, cache setup (`DSPY_CACHEDIR`, `DSPY_CACHE_LIMIT`).
- **Signatures & typed I/O**: `dspy/signatures/`
  - `Signature`, `InputField`, `OutputField`, signature parsing/validation.
- **Adapters & structured output parsing**: `dspy/adapters/`
  - Adapters like `ChatAdapter`, `JSONAdapter`, `XMLAdapter`, tool calling types, parsing utilities.
- **Primitives**: `dspy/primitives/`
  - `Module`, `Example`, `Prediction`, tool and media primitives, interpreters (sandbox + local).
- **Inference modules ("predictors")**: `dspy/predict/`
  - `Predict`, `ChainOfThought`, `ReAct`, `ProgramOfThought`, `RLM`, etc.
- **Optimizers / "teleprompters"**: `dspy/teleprompt/`
  - Compile/optimize programs (e.g. `BootstrapFewShot`, `GEPA`, `COPRO`, ...).
- **Evaluation**: `dspy/evaluate/`
  - `Evaluate` + metrics.
- **Retrieval**: `dspy/retrievers/` and `dspy/dsp/colbertv2.py`
  - `Embeddings` retriever, `ColBERTv2` client.
- **Cross-cutting utilities**: `dspy/utils/`
  - Callbacks, async/sync helpers, parallel execution, caching/logging utilities, tool conversions (LangChain, MCP).
- **Streaming utilities**: `dspy/streaming/`
  - `streamify(...)` wrapper for incremental output streaming + status messages; `StreamListener` plumbing.
- **Experimental APIs**: `dspy/experimental/`
  - Thin re-export surface for experimental types (e.g., `Citations`, `Document`).
- **Proposers**: `dspy/propose/`
  - Base proposer interface + implementations (e.g., grounded proposer).
- **Datasets**: `dspy/datasets/`
  - Built-in dataset loaders/wrappers (e.g., `gsm8k`, `hotpotqa`, `math`, `colors`, `alfworld`).
  - Scriptable dataset smoke-test: `dspy/datasets/hotpotqa.py` (has `__main__`).

Non-library directories:

- **CI + GitHub metadata**: `.github/` (workflows, issue/PR templates, helper scripts)
- **Tests**: `tests/` (unit/integration-style tests; see "Test map" below)
- **Docs**: `docs/` (MkDocs site + generators; see "Build/tooling map" below)

## 3) Symbol Map (Top-Level, "Where Do I Find X?")

The top-level package (`import dspy`) re-exports most of its public API from submodules; these are the "go-to" definitions.

| Symbol | What it is | Defined in |
| --- | --- | --- |
| `dspy.configure`, `dspy.context`, `dspy.load_settings`, `dspy.settings` | Global config + thread-local overrides | `dspy/dsp/utils/settings.py` (aliased in `dspy/__init__.py`) |
| `dspy.configure_cache`, `dspy.cache` | DSPy cache setup + singleton | `dspy/clients/__init__.py` (configure_cache + DSPY_CACHE), `dspy/__init__.py` (aliases `cache = DSPY_CACHE`) |
| `dspy.LM` | LiteLLM-backed LLM client | `dspy/clients/lm.py` |
| `dspy.CodexCLI` | LLM client that shells out to local `codex exec` | `dspy/clients/codex_cli.py` |
| `dspy.Embedder` | Embeddings wrapper (LiteLLM or custom fn) | `dspy/clients/embedding.py` |
| `dspy.BaseLM`, `dspy.Provider`, `dspy.TrainingJob`, `dspy.inspect_history` | Client/provider base types + helpers | `dspy/clients/base_lm.py`, `dspy/clients/provider.py`, `dspy/clients/__init__.py` |
| `dspy.enable_litellm_logging`, `dspy.disable_litellm_logging` | LiteLLM logging controls | `dspy/clients/__init__.py` |
| `dspy.Signature` | Typed input/output contract | `dspy/signatures/signature.py` |
| `dspy.InputField`, `dspy.OutputField` | Signature field descriptors | `dspy/signatures/field.py` |
| `dspy.Module`, `dspy.BaseModule` | Base compositional units | `dspy/primitives/module.py`, `dspy/primitives/base_module.py` |
| `dspy.Example`, `dspy.Prediction`, `dspy.Completions` | Data containers / completion wrapper | `dspy/primitives/example.py`, `dspy/primitives/prediction.py` |
| `dspy.CodeInterpreter`, `dspy.CodeInterpreterError`, `dspy.FinalOutput` | Interpreter protocol + errors + submit sentinel | `dspy/primitives/code_interpreter.py` |
| `dspy.RLM` | Recursive Language Model (REPL + sub-LLMs) | `dspy/predict/rlm.py` |
| `dspy.PythonInterpreter` | Sandboxed interpreter (default for RLM) | `dspy/primitives/python_interpreter.py` |
| `dspy.primitives.local_interpreter.LocalInterpreter` | UNSANDBOXED host-Python interpreter (not re-exported at `dspy.*`) | `dspy/primitives/local_interpreter.py` |
| `dspy.Adapter`, `dspy.ChatAdapter`, `dspy.JSONAdapter`, `dspy.XMLAdapter`, `dspy.TwoStepAdapter` | Structured output / tool-calling adapters | `dspy/adapters/base.py`, `dspy/adapters/chat_adapter.py`, `dspy/adapters/json_adapter.py`, `dspy/adapters/xml_adapter.py`, `dspy/adapters/two_step_adapter.py` |
| `dspy.ToolCalls`, `dspy.Type`, `dspy.History`, `dspy.Image`, `dspy.Audio`, `dspy.File`, `dspy.Code`, `dspy.Reasoning`, `dspy.Tool` | Tool-call + multimodal payload wrapper types | `dspy/adapters/types/__init__.py`, `dspy/adapters/types/tool.py` |
| `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.ProgramOfThought` | Common predictor modules | `dspy/predict/*` |
| `dspy.CodeAct`, `dspy.Refine`, `dspy.BestOfN`, `dspy.KNN`, `dspy.MultiChainComparison`, `dspy.Parallel`, `dspy.majority` | Additional predictors/utilities | `dspy/predict/*` |
| `dspy.ColBERTv2` | ColBERTv2 retriever client | `dspy/dsp/colbertv2.py` |
| `dspy.Embeddings`, `dspy.Retrieve` | Retrieval modules | `dspy/retrievers/embeddings.py`, `dspy/retrievers/retrieve.py` |
| `dspy.Evaluate` | Evaluation runner | `dspy/evaluate/evaluate.py` |
| `dspy.BootstrapFewShot`, `dspy.BootstrapFewShotWithRandomSearch`, `dspy.BootstrapRS` | Bootstrap optimizers (+ alias) | `dspy/teleprompt/bootstrap.py`, `dspy/teleprompt/random_search.py`, `dspy/__init__.py` |
| `dspy.MIPROv2`, `dspy.COPRO`, `dspy.SIMBA`, `dspy.GEPA` | Other optimizers | `dspy/teleprompt/mipro_optimizer_v2.py`, `dspy/teleprompt/copro_optimizer.py`, `dspy/teleprompt/simba.py`, `dspy/teleprompt/gepa/gepa.py` |
| `dspy.load`, `dspy.streamify`, `dspy.track_usage`, `dspy.asyncify`, `dspy.syncify` | Save/load + streaming + usage tracking + async helpers | `dspy/utils/saving.py`, `dspy/streaming/streamify.py`, `dspy/utils/usage_tracker.py`, `dspy/utils/asyncify.py`, `dspy/utils/syncify.py` |

Note: `dspy.Tool` is the adapter/type-system tool wrapper (`dspy/adapters/types/tool.py`). `ReAct` also defines a different `Tool`, but `dspy.Tool` at top-level is the adapter one (it overwrites the predictor export).

## 4) Dependency Edges (How The Pieces Fit)

Typical "plain DSPy" path:

- User code constructs **signatures** (`dspy.Signature` or `"a, b -> c"` string)
- User composes **modules** (`dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, ...)
- Modules call the configured **LM** from settings:
  - `dspy.configure(lm=...)` sets `dspy.settings.lm`
  - `dspy.Predict.forward(...)` uses `settings.lm` + an adapter to generate structured outputs
- Outputs become `Prediction` objects with typed fields
- Optional: an **optimizer** in `dspy.teleprompt.*` "compiles" a program by repeatedly evaluating it and updating demonstrations/prompts/weights

Concrete call graph (sync, common path):

- `program(**inputs)` -> `dspy.primitives.module.Module.__call__` -> `program.forward(...)`
- `dspy.Predict.forward`:
  - `adapter = dspy.settings.adapter or dspy.ChatAdapter()`
  - `completions = adapter(lm=dspy.settings.lm, signature=..., demos=..., inputs=..., lm_kwargs=...)`
  - `Prediction.from_completions(...)`
  - appends `(predictor, inputs, prediction)` to `dspy.settings.trace` (when enabled)
- Adapter pipeline (`dspy/adapters/base.py`):
  - `Adapter.format(...)` builds `messages=[system + demos + history + user]`
  - `lm(messages=messages, **lm_kwargs)` (i.e., `dspy.clients.base_lm.BaseLM.__call__`)
  - `Adapter.parse(...)` turns text into a dict matching output fields
- `ChatAdapter` fallback: retries via `JSONAdapter` on parse/format errors (except context-window exceeded)
- Tool calling (native function calling): when enabled, `Adapter._call_preprocess` injects `tools=` if the signature includes `dspy.Tool` inputs and a `ToolCalls` output.

Cache edges (where memoization actually hooks in this repo):

- Shared caching hook: `dspy.clients.cache.request_cache(...)` reads/writes the global `dspy.cache`.
- Used by:
  - `dspy.clients.lm.LM` (provider calls via LiteLLM) when `LM(cache=True)` (default)
  - `dspy.clients.embedding.Embedder` (embedding batches)
  - `dspy.dsp.colbertv2.ColBERTv2` (cached HTTP requests)

Retrievers (RM) call graph:

- `dspy.Retrieve.forward(query, k=...)` -> `dspy.settings.rm(query, k=k, **kwargs)` -> returns `Prediction(passages=[...])`

Teleprompt <-> Evaluate loop (how "compile" runs programs):

- Many `dspy.teleprompt.*.compile(...)` implementations score candidate programs via `dspy.Evaluate(devset=..., metric=...)(program)`.
- Trace-dependent optimizers run under `with dspy.context(trace=[]): ...` and read `dspy.settings.trace`, populated by `Predict._forward_postprocess`.
- `BootstrapFewShot` bypasses caches across rounds via `lm.copy(rollout_id=round_idx, temperature=1.0)`.

RLM path (repo-specific highlight):

- `dspy.RLM(...)` runs a loop where the model writes Python code against a **CodeInterpreter**
- The interpreter exposes `llm_query*` helpers that call a sub-LM (default: `dspy.settings.lm`)
- Default interpreter is `PythonInterpreter` (sandbox); `LocalInterpreter` is explicitly UNSANDBOXED

RLM / Interpreter / sub-LMs (repo-grounded call graph):

- `dspy.RLM.__init__` builds two internal predictors:
  - `self.generate_action = dspy.Predict(action_sig)` (LLM writes next REPL code)
  - `self.extract = dspy.Predict(extract_sig)` (fallback: extract final outputs from trajectory)
- `RLM.forward(**inputs)`:
  - registers tools (`llm_query`, `llm_query_batched`, `llm_query_with_media`, `budget`, plus user tools)
  - selects interpreter: default `PythonInterpreter`, or user-provided `interpreter=...`
  - loops until `SUBMIT(...)` returns a `FinalOutput`; otherwise may fall back to `self.extract(...)`
- Sub-LM calls from inside the sandbox:
  - `llm_query(prompt, model=...)` resolves `sub_lms[model]` -> `sub_lm` -> `dspy.settings.lm`
  - if `depth < max_depth - 1`, `llm_query` spawns a child `RLM("prompt -> response", depth+1)`

Codex CLI client + `rlm` CLI path (repo-specific highlight):

- `dspy.CodexCLI` is an `LM` implementation that runs `codex exec ...` per request
- `rlm` CLI (`dspy/cli/rlm.py`) wires together:
  - a `CodexCLI(...)` backend
  - an `RLM(...)` module
  - optional built-in "workspace file tools" in RLM mode
  - a curses TUI for interactive sessions

CodexCLI + `rlm` CLI wiring (repo-grounded):

- `dspy/cli/rlm.py` constructs `lm = dspy.CodexCLI(...)` and calls `dspy.configure(lm=lm)`.
- In `--mode rlm`, it defines workspace-rooted file tools (`workspace_root`, `list_dir`, `read_text_file`) and passes them to `dspy.RLM(tools=[...])`.
- Interpreter selection:
  - default: sandboxed `PythonInterpreter`
  - `--unsafe-local-interpreter`: UNSANDBOXED `LocalInterpreter` (host Python `exec`)

## 5) Config + Runtime

DSPy configuration is centralized in `dspy.settings`:

- Set global defaults: `dspy.configure(lm=..., adapter=..., rm=..., ...)`
- Override per-thread/per-block: `with dspy.context(...): ...`

Cache-related env vars:

- `DSPY_CACHEDIR`: disk cache directory (default `~/.dspy_cache`)
- `DSPY_CACHE_LIMIT`: disk cache size limit in bytes (default `3e10`)

Fine-tuning env vars:

- `DSPY_FINETUNEDIR`: fine-tuning artifacts dir (default `<DSPY_CACHEDIR>/finetune`)

Adapter/runtime env vars:

- `DSPY_JSON_ADAPTER_REPAIR`: if set to `1/true/yes`, enables JSON-repair pass in `dspy.JSONAdapter`

RLM CLI state + editor env vars:

- `DSPY_RLM_HOME` / `RLM_HOME`: override `rlm` CLI state dir (default `~/.dspy/rlm`)
- `VISUAL` / `EDITOR`: editor command used by `rlm` TUI "/edit" (fallback `vi` / `notepad`)

Sandbox env vars:

- `DENO_DIR`: Deno cache dir used by `PythonInterpreter` (also influences its stable sandbox cwd)

LiteLLM runtime env vars:

- `LITELLM_LOCAL_MODEL_COST_MAP`: set to `"True"` on import if missing (LiteLLM local model cost map)

Provider auth env vars are provider-specific; see the docs' "Getting Started" page:

- `docs/docs/index.md` (shows `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, Databricks env vars, etc.)
- Databricks RM defaults (used by `dspy/retrievers/databricks_rm.py`): `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`

Security-sensitive runtime knobs (RLM/Codex):

- `dspy.CodexCLI(..., sandbox=..., ask_for_approval=...)` controls Codex CLI sandbox + approvals
- `rlm` CLI flags in `dspy/cli/rlm.py`:
  - `--codex-sandbox`, `--codex-enable-shell-tool`, `--codex-approval`, `--unsafe-local-interpreter`, `--unsafe-local-subcalls`
  - Other useful knobs: `--codex-trust-level`, `--codex-prompt-mode`, `--codex-command`, `--cwd`, `--model`, `--reasoning-effort`, `--timeout-seconds`, `--max-iterations`, `--max-llm-calls`, `--max-time`, `--max-tokens`, `--max-cost`

Sandbox permission knobs (Python API):

- `dspy.PythonInterpreter(..., enable_read_paths=..., enable_write_paths=..., enable_env_vars=..., enable_network_access=..., sync_files=...)` governs sandbox access (default is deny host FS/network/env).

RLM CLI persistent runtime files:

- `rlm` persists history/sessions under `~/.dspy/rlm/` (e.g. `history/input_<mode>.jsonl`, `sessions/<id>.json`); root overridden by `DSPY_RLM_HOME` / `RLM_HOME`.

## 6) Test Map

Where tests live:

- Unit tests: `tests/` (organized by subsystem: `tests/clients`, `tests/predict`, `tests/teleprompt`, ...)
- Reliability suite: `tests/reliability/` (separate harness/config)
- Docs link checks: `tests/docs/test_mkdocs_links.py`
  - Reliability config: `tests/reliability/reliability_conf.yaml`

How to run tests (common):

- With uv (CI-style):
  - `uv venv .venv`
  - `uv sync --dev -p .venv --extra dev`
  - `uv run -p .venv pytest -vv tests/`

Skip-gated markers:

- Tests marked `extra`, `deno`, `llm_call`, or `reliability` are skipped unless you also pass the matching flag: `--extra`, `--deno`, `--llm_call`, `--reliability` (even if you select them via `-m`).

Notable CI test modes (`.github/workflows/run_tests.yml`):

- "Extra" deps + Deno-marked tests: installs `--extra test_extras` and runs `pytest -m 'extra or deno' --extra --deno`
- Real-LM tests: `LM_FOR_TEST=... pytest -m llm_call --llm_call` (CI uses Ollama + Docker)

Common commands (CI-matching):

- Base tests:
  - `uv run -p .venv pytest -vv tests/`
- Extra + Deno tests:
  - `uv sync -p .venv --extra dev --extra test_extras`
  - `uv run -p .venv pytest tests/ -m 'extra or deno' --extra --deno`
- Real-LM tests:
  - `LM_FOR_TEST=ollama/llama3.2:3b uv run -p .venv pytest -m llm_call --llm_call -vv --durations=5 tests/`
- Reliability suite (not run in CI by default):
  - `uv run -p .venv pytest tests/reliability -m reliability --reliability`

Runnables / harness helpers:

- Reliability test-case generator: `python -m tests.reliability.generate -d <dst_path> ...` (entrypoint `tests/reliability/generate/__main__.py`)
- MCP test server: `tests/utils/resources/mcp_server.py` (runs `mcp.run()` when executed directly)

## 7) Build / Tooling Map

Python packaging:

- Build system: `pyproject.toml` (setuptools backend)
- Local build: `python -m build` (see CI job `build_package` in `.github/workflows/run_tests.yml`)

Dev environment:

- Recommended: uv (see "Environment Setup" in `CONTRIBUTING.md`)
  - `uv venv --python 3.10`
  - `uv sync --extra dev`

Lockfile:

- `uv.lock` is committed and used by CI caching; update it when changing `pyproject.toml` deps/extras.

Lint/format:

- Ruff config: `pyproject.toml` under `[tool.ruff]`
- Pre-commit hooks: `.pre-commit-config.yaml`
- CI lint/test/build gate: primarily `.github/workflows/run_tests.yml` (push + PR)
- Manual-only workflow: `.github/workflows/precommits_check.yml` (`workflow_dispatch` only)

Docs build:

- `cd docs && pip install -r requirements.txt`
- API docs generation: `python scripts/generate_api_docs.py` + `python scripts/generate_api_summary.py`
- Build: `mkdocs build` (or `mkdocs serve` for local dev)
- CI: `.github/workflows/docs-push.yml` builds on PRs touching `docs/**` and (on `main` in `stanfordnlp/dspy`) pushes the `docs/` subtree to the separate docs repo

Release process (tags):

- Pushing a tag triggers `.github/workflows/build_and_release.yml`:
  - publishes `dspy-ai-test` to TestPyPI, then publishes `dspy` and `dspy-ai` to PyPI
  - uses `sed` against `pyproject.toml` markers, so `name="..."` / `version="..."` formatting is intentionally strict (no spaces around `=`)
