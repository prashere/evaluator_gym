# Evaluator Gym

A verifiable evaluation environment for AI agents doing AP invoice reconciliation. An agent reads case documents (and optional tools), returns structured JSON, and gets a score traceable to a written rule set. Ground truth always comes from a Python reference implementation, never from an LLM.

**Live dashboard:** [https://prashere.github.io/evaluator_gym/](https://prashere.github.io/evaluator_gym/)  
**Live sandbox:** [https://evaluator-gym-sandbox.onrender.com](https://evaluator-gym-sandbox.onrender.com)  

Reviewer write-ups (eval, failures, edge cases, assumptions, RL training, AI usage) are in `[submission_docs/](submission_docs/)`.

---

## 1. Clone and install

```bash
git clone https://github.com/prashere/evaluator_gym.git
cd evaluator_gym

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Copy environment variables for local eval (optional if you only use the live sandbox and dashboard):

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY (and optionally OPENAI_API_KEY)
```

**Sanity check** (no API key required):

```bash
python scripts/validate_tasks.py
pytest -q
python -m dashboard.build
```

If these pass, the task schema, reference ground truth, rubric, and dashboard build pipeline are consistent on your machine.

---

## 2. Live sandbox (no local server)

The sandbox is a hosted API where you create a run, call tools to read documents, submit JSON answers, and read scores. It never returns ground truth.

**Fastest path:** from the repo root, with only `curl` and `python3`:

```bash
export SANDBOX_URL=https://evaluator-gym-sandbox.onrender.com
chmod +x scripts/sandbox_reviewer_roundtrip.sh
./scripts/sandbox_reviewer_roundtrip.sh
```

Success ends with `Reviewer round trip PASSED`. The script uses a stub HOLD answer on one tier-2 task, so the score will be low. That is expected. It proves the API path works.

**Customize tier or decision:**

```bash
SANDBOX_TIER=3 SANDBOX_DECISION=ESCALATE ./scripts/sandbox_reviewer_roundtrip.sh
```

**Full manual walkthrough** (every curl step, tier shapes, cold-start notes): `[sandbox_reviewer_guide.md](sandbox_reviewer_guide.md)`.


| Tier | What it tests                       | Answer shape                                                          |
| ---- | ----------------------------------- | --------------------------------------------------------------------- |
| 1    | Field lookup from context           | `{"field_name": "value"}`                                             |
| 2    | Full reconciliation                 | `{"decision": "APPROVE|HOLD|ESCALATE", "evidence_set": ["TAG", ...]}` |
| 3    | Traps (missing or conflicting data) | Same as tier 2; often HOLD or ESCALATE                                |


Tool mode requires a `completion` transcript on submit so the rubric can see which documents were read.

---

## 3. Dashboard

The dashboard is a static site built from **committed** JSON under `results/`. It does not call Groq or re-run eval on deploy.

**Use the deployed site:** open [https://prashere.github.io/evaluator_gym/](https://prashere.github.io/evaluator_gym/)  
Default eval run: `golden-matrix-p3` (12 fixed seed tasks, three Groq models, three rollouts each).

**Rebuild locally** after you change or add results:

```bash
./scripts/refresh_dashboard.sh                  # uses dashboard/default_run_id.txt
./scripts/refresh_dashboard.sh golden-matrix-p3
WITH_BASELINES=1 ./scripts/refresh_dashboard.sh golden-matrix-p3   # include adversarial baselines
```

Preview:

```bash
cd dashboard/dist && python -m http.server 8765
# Open http://localhost:8765
```

The RL tab reads figures from `results/training/phase07-v1/` and `results/training/phase07-v2/`.

---

## 4. Run eval locally (optional)

Eval needs a Groq API key in `.env`. Results land in `results/<model-slug>/<run-id>/`.

**Golden matrix (same 12 tasks as the dashboard default):**

```bash
./scripts/run_golden_matrix_p3.sh
```

**Single model, resume after a partial run:**

```bash
python -m evaluator_gym.eval \
  --model groq/gpt-oss-20b \
  --benchmark golden-matrix-p3 \
  --resume results/groq-gpt-oss-20b/golden-matrix-p3
```

Each run directory contains `config.json`, `transcript.jsonl`, `metrics.json`, and `scores.json`. Provider failures (rate limits, empty responses) are tagged separately from rubric scores.

---

## 5. Run sandbox locally (optional)

```bash
python -m evaluator_gym.sandbox
# Default: http://localhost:8080
```

In another terminal:

```bash
SANDBOX_URL=http://localhost:8080 ./scripts/sandbox_quickstart.sh
```

Local sandbox uses the same rubric as eval. Deploy config lives in `render.yaml` for Render.

---

## 7. Architecture

This section explains how the pieces connect. I kept one scoring brain and one ground-truth path so eval, sandbox, and training stay comparable.

### 7.1 End-to-end flow

```text
rules/v1.0.0/RULES.md          (business rules, human-readable)
        |
        v
reference/                     (Python engine: decisions + evidence tags)
        |
        v
generator/ + tasks/seed/       (tasks + stored ground_truth, deterministic variants)
        |
        v
environment + parser + tools   (what the agent sees; no hidden answers)
        |
        v
rubric/ score_task()           (one scorer for eval, sandbox, training)
        |
        +-------- eval  -> results/     (local harness, committed JSON)
        +-------- sandbox -> live API  (scores only, no GT leakage)
        +-------- training -> RL loop  (reward + KL to frozen reference)
        |
        v
dashboard/build.py             (static leaderboard from committed results/)
```

**Why this shape.** Invoice reconciliation needs auditable rules, reproducible tasks, and scores that cite those rules. If ground truth came from model output, or if sandbox used a different scorer than eval, reviewers could not trust the numbers. I enforced a single reference path and a single rubric entry point.

### 7.2 Major components


| Component        | Location                                                     | Role                                                    |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| Rule set         | `rules/v1.0.0/RULES.md`                                      | Domain truth for tiers 2 and 3                          |
| Reference engine | `src/evaluator_gym/reference/`                               | Computes `decision` + `evidence_set` from case JSON     |
| Seed tasks       | `tasks/seed/` (23 tasks)                                     | Hand-checked benchmark instances with stored GT         |
| Generator        | `src/evaluator_gym/generator/`                               | Seeded procedural variants; GT from reference only      |
| Task schemas     | `tasks/task.schema.json`, `tasks/agent_response.schema.json` | Contracts for tasks and agent JSON                      |
| Environment      | `src/evaluator_gym/environment.py`                           | `load_environment()` → verifiers SingleTurn or Tool env |
| Tools            | `src/evaluator_gym/tools.py`                                 | `read_document`, `read_policy`, `python_calc`           |
| Parser           | `src/evaluator_gym/parser.py`                                | Validates agent JSON; classifies parse failures         |
| Rubric           | `src/evaluator_gym/rubric/`                                  | Weighted components + gates → `final_reward`            |
| Eval harness     | `src/evaluator_gym/eval/`                                    | Model rollouts → `results/`                             |
| Dashboard        | `dashboard/`                                                 | Offline build from `results/`                           |
| Sandbox          | `src/evaluator_gym/sandbox/`                                 | FastAPI service; public score breakdown only            |
| Training         | `src/evaluator_gym/training/`, notebooks                     | Colab QLoRA + group-relative policy gradient            |


CI runs `scripts/validate_tasks.py` (schema, GT recomputation, leakage checks) and pytest, including hand-audited fidelity cases that are independent of the reference loop.

### 7.3 Task tiers


| Tier                 | Skill                                           | Agent output                                      |
| -------------------- | ----------------------------------------------- | ------------------------------------------------- |
| Tier 1 (retrieval)   | Read one fact from provided context             | Field JSON, e.g. `{"vendor_id": "..."}`           |
| Tier 2 (computation) | Multi-step reconcile on complete evidence       | `{decision, evidence_set}`                        |
| Tier 3 (traps)       | Missing, conflicting, or non-determinable input | `{decision, evidence_set}`; HOLD or ESCALATE only |


Tier 1 prompts omit full policy text so the task stays lookup-only. Tiers 2 and 3 include `RULES.md` (single-turn) or `read_policy()` (tool mode).

### 7.4 Ground truth

Ground truth is always `reference.compute_ground_truth(task)`. The generator emits tasks and calls the reference; it does not invent tags or decisions. Seed files store GT for audit, and CI recomputes on every run. The environment package does not import `reference/`, so prompts and tools cannot accidentally embed answers.

### 7.5 Scoring (rubric 0.1.2)

Single entry point: `score_task()` in `src/evaluator_gym/rubric/build.py`.

**Tier 1 (retrieval):** 100% field accuracy.

**Tiers 2 and 3 (single-turn eval):**


| Component              | Weight |
| ---------------------- | ------ |
| `decision_correct`     | 0.45   |
| `evidence_f1`          | 0.35   |
| `precedence_coherence` | 0.10   |
| `calibration`          | 0.10   |
| `tag_support_judge`    | 0.00   |


**Tool mode** shifts 10% to `tag_support_judge` (LLM checks tag support against documents read). Judge runs only when a tool transcript exists.

**Gates** (after adversarial testing): wrong decision capped at 0.10; spurious tags on clean APPROVE capped at 0.45; zero tag overlap with correct decision capped at 0.10; tier-1 wrong field capped at 0.30. Infrastructure failures (parse error, provider timeout) are not scored as zero reward without a `failure_class`.

### 7.6 Three runtime surfaces


| Surface   | Runs where                 | Output                                             |
| --------- | -------------------------- | -------------------------------------------------- |
| Eval      | Your machine + API keys    | `results/<model>/<run-id>/` JSON, committed to git |
| Dashboard | Static host (GitHub Pages) | Reads committed `results/` only                    |
| Sandbox   | Render (live URL above)    | Per-run scores + rule clause IDs; no ground truth  |
| Training  | Colab + Drive checkpoints  | `results/training/` metrics and figures in git     |


Eval never runs on deploy. That keeps secrets off Pages and makes the dashboard reproducible from the repo alone.

---

## 8. Repository layout

```text
rules/                  Versioned RULES.md
tasks/                  Schemas, seed tasks, eval_golden.json
src/evaluator_gym/      Core package (reference, env, rubric, eval, sandbox, training)
scripts/                validate_tasks, eval matrices, dashboard refresh, sandbox QA
results/                Committed eval and baseline artifacts
results/training/       RL metrics and figures
dashboard/              Static site builder and dist output
submission_docs/        Reviewer-facing reports (deliverables 6 to 12)
notebooks/              RL training notebook
tests/                  Unit, integration, fidelity, adversarial tests
sandbox_reviewer_guide.md   Detailed live sandbox instructions
```

---

## 9. Submission documents


| Document                   | Path                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- |
| Eval report                | `[submission_docs/eval_report.md](submission_docs/eval_report.md)`                     |
| Failure analysis           | `[submission_docs/failure_analysis.md](submission_docs/failure_analysis.md)`           |
| Edge cases                 | `[submission_docs/edge_cases.md](submission_docs/edge_cases.md)`                       |
| Assumptions and trade-offs | `[submission_docs/assumptions_tradeoffs.md](submission_docs/assumptions_tradeoffs.md)` |
| RL training report         | `[submission_docs/rl_training_report.md](submission_docs/rl_training_report.md)`       |
| AI usage disclosure        | `[submission_docs/ai_usage.md](submission_docs/ai_usage.md)`                           |


---

