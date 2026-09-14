# AI Usage Disclosure

This document explains how I used AI tools while building the gym. I used them for planning, search, and implementation speed. I did not use them to invent domain rules, ground truth, or evaluation numbers. Every score in this repo comes from the reference implementation and rubric code, not from model-generated answers passed off as truth.

---

## How I used each tool

| Tool | I used it for | I did not use it for |
|---|---|---|
| **ChatGPT** | Breaking down the assignment into sections and clarifying early questions; comparing payroll vs AP reconciliation with real professional case examples; finding Prime Intellect GitHub issues and related threads when I needed v0 vs v1 context or edge-case pointers | Writing `RULES.md`, computing ground truth, or deciding final rubric weights |
| **Claude** | Second opinion on domain choice (same payroll vs AP question), resource pointers, and architecture or pipeline feedback before I split work into sprints | Final authority on business semantics when the written spec was silent |
| **Cursor (coding agent)** | Most implementation: environment, rubric wiring, eval harness, dashboard, sandbox, RL notebook plumbing, unit test drafts, submission docs, and `.cursor/rules/` agent guidance files | Unreviewed merges; I read and edited generated code and tests before treating them as done |

---

## Flow in order

**1. Understanding the assignment.** I pasted the assignment into ChatGPT and asked it to break the doc into sections and surface questions I should answer before coding. That gave me a reading order, not a substitute for reading the spec myself.

**2. Choosing the domain.** I was unsure between payroll and AP invoice reconciliation. I asked ChatGPT and Claude for examples of work real professionals handle in each area, plus pointers to public resources. I compared the answers myself and picked AP reconciliation because the rule structure and evidence trail fit a verifiable benchmark better.

**3. Research on Prime Intellect / verifiers.** Skimming every GitHub issue to see how v1 differed from v0 was slow. I used ChatGPT as a search assistant: I described the edge case or API question from the assignment, got issue numbers or keywords, then opened those threads and read the source myself.

**4. Architecture and sprints.** I drafted my own phase order, then asked Claude for gaps or ordering risks. I turned the result into small sprints (tasks, generator, rubric, eval, and so on) and adjusted when the spec or tests disagreed with the suggestion.

**5. Development in Cursor.** Cursor wrote most of the code and first drafts of tests. I configured agent rules in `.cursor/rules/` so the agent knew invariants (ground truth from reference only, one rubric everywhere, no sandbox leakage). I stepped through unit tests and spec checks rather than trusting green output alone.

**6. Documentation.** Cursor helped draft reports under `submission_docs/` (eval, failure analysis, edge cases, assumptions, RL training, this file). I edited for accuracy against committed artifacts in `results/` and `results/training/`.

---

## What stayed human-owned

These items were never delegated to an LLM as source of truth:

- **`rules/v1.0.0/RULES.md`** and what each §13 tag means in reconciliation
- **Ground truth** via `src/evaluator_gym/reference/` (Python only)
- **Rubric semantics and gates** after adversarial baseline review
- **Benchmark task selection** (seed set, golden-matrix-p3 subset, train vs held-out seeds)
- **Reported metrics** (eval and RL numbers come from local runs, not invented in chat)

When the spec was ambiguous, I stopped and chose explicitly (often documented in `submission_docs/assumptions_tradeoffs.md` or asked in implementation policy) instead of letting the model guess domain meaning.

---

## How I checked AI output

- **Code:** pytest, CI, and manual reads on reward, parser, and sandbox paths
- **Ground truth:** reference tests and seed recomputation in CI
- **Eval and RL claims:** only numbers copied from `results/` and `results/training/` JSON
- **Docs:** cross-check against artifacts before submit

If a tool suggestion conflicted with `RULES.md` or the assignment tier definitions, I changed the code or docs, not the rule set, unless I had an explicit product decision recorded elsewhere.
