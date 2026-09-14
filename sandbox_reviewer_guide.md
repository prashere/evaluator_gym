# Sandbox Reviewer Guide

This guide shows how to test the **live** sandbox API without running a local server. You only need `curl` and `python3`. No API keys are required on your machine.

**Live URL:** https://evaluator-gym-sandbox.onrender.com

---

## What a round trip is

```text
1. Create run     POST /v1/runs              → run_id + task prompts
2. Use tools      POST .../tasks/{id}/tools  → read invoices, policy, etc.
3. Submit answers POST .../submit            → your JSON answer(s)
4. Read results   GET  .../runs/{run_id}     → scores + per-rule breakdown
```

The sandbox never returns ground truth. It scores your submit with the same rubric as eval and returns public fields only: total score, component scores, and rule clause IDs (for example `"§12"`). It does not return full rule text or lists of missing or extra tags.

The script `scripts/sandbox_reviewer_roundtrip.sh` runs all four steps with a stub tier-2 answer. It checks that the pipeline works. It is not meant to produce a high score.

---

## Configuration

Copy this block into your terminal and change values as needed.

```bash
export SANDBOX_URL=https://evaluator-gym-sandbox.onrender.com

export SANDBOX_TIER=2          # 1 | 2 | 3 | all
export SANDBOX_N=1             # 1 to 10 tasks per run

export SANDBOX_READ_DOCS=case_context.json,policy
export SANDBOX_DECISION=HOLD   # APPROVE | HOLD | ESCALATE
export SANDBOX_EVIDENCE='[]'   # JSON array of tags, e.g. '["INVOICE_INCOMPLETE"]'

export SANDBOX_TIER1_ANSWER='{"vendor_id": "reviewer-test"}'

export SANDBOX_VERBOSE=0
export SANDBOX_CURL_TIMEOUT=90
```

| Tier | Skill | Answer shape | Tools |
|---|---|---|---|
| 1 | Retrieval | One field, e.g. `{"vendor_id": "..."}` | `read_document`, `python_calc` (no `read_policy`) |
| 2 | Computation | `{"decision": "...", "evidence_set": [...]}` | All three tools |
| 3 | Traps | Same as tier 2; often HOLD or ESCALATE | All three tools |
| all | Mixed | One answer per task, shape matches each task tier | Per task |

When `SANDBOX_TIER=all` and `SANDBOX_N=3`, the server returns one task per tier (1+1+1), not three random tier-2 tasks.

---

## Automated round trip (fastest)

```bash
cd evaluator_gym
chmod +x scripts/sandbox_reviewer_roundtrip.sh
./scripts/sandbox_reviewer_roundtrip.sh
```

Examples:

```bash
SANDBOX_TIER=3 SANDBOX_DECISION=ESCALATE ./scripts/sandbox_reviewer_roundtrip.sh
SANDBOX_TIER=all SANDBOX_N=3 ./scripts/sandbox_reviewer_roundtrip.sh
SANDBOX_VERBOSE=1 ./scripts/sandbox_reviewer_roundtrip.sh
./scripts/sandbox_reviewer_roundtrip.sh --help
```

Success: the last line is `Reviewer round trip PASSED against ...`

---

## Manual steps

### Step 0: Wake the service

Free-tier hosts can cold-start. Retry if the first call is slow.

```bash
curl -s --max-time "${SANDBOX_CURL_TIMEOUT:-90}" \
  "${SANDBOX_URL}/health" | python3 -m json.tool
```

Expect `"status": "ok"`.

### Step 1: Create a run

```bash
curl -s --max-time "${SANDBOX_CURL_TIMEOUT:-90}" \
  -X POST "${SANDBOX_URL}/v1/runs" \
  -H "Content-Type: application/json" \
  -d "{\"tier\": \"${SANDBOX_TIER:-2}\", \"n\": ${SANDBOX_N:-1}}" \
  | python3 -m json.tool
```

Save `run_id` and task `id` from the response:

```bash
export RUN_ID="paste_run_id_here"
export TASK_ID="paste_task_id_here"
```

Check that the JSON has no `ground_truth` field.

### Step 2: Call tools

Read case context:

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "read_document", "arguments": {"doc_id": "case_context.json"}}' \
  | python3 -m json.tool
```

Common document ids:

| `doc_id` | Content |
|---|---|
| `case_context.json` | Case metadata |
| `invoice.json` | Invoice lines |
| `purchase_order.json` | PO |
| `goods_receipt.json` | Receipt |
| `vendor_record.json` | Vendor master |
| `approval_evidence.json` | Approval trail |

Read policy (tier 2 and 3 only):

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "read_policy", "arguments": {}}' \
  | python3 -m json.tool
```

Calculator:

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "python_calc", "arguments": {"expression": "100 + 25.50"}}' \
  | python3 -m json.tool
```

### Step 3: Submit answers

Submit exactly one answer per task id from Step 1.

**Tier 2 or 3 (single task):**

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"answers\": [{
      \"id\": \"${TASK_ID}\",
      \"answer\": {
        \"decision\": \"${SANDBOX_DECISION:-HOLD}\",
        \"evidence_set\": ${SANDBOX_EVIDENCE:-[]}
      },
      \"completion\": [{\"role\": \"tool\", \"content\": \"reviewer manual test\"}]
    }]
  }" | python3 -m json.tool
```

Tool mode requires a `completion` transcript so the rubric can see which documents were read.

Valid evidence tags are listed in `tasks/agent_response.schema.json`.

**Tier 1 (field lookup):**

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"answers\": [{
      \"id\": \"${TASK_ID}\",
      \"answer\": ${SANDBOX_TIER1_ANSWER:-{\"vendor_id\": \"reviewer-test\"}},
      \"completion\": []
    }]
  }" | python3 -m json.tool
```

Read the task prompt first. Tier 1 expects the specific field named in the prompt (for example `vendor_id` or `ruleset_version`). A placeholder value proves the API works but may score poorly if the field or value is wrong.

**Multiple tasks (`tier=all`, `n=3`):**

```bash
curl -s -X POST "${SANDBOX_URL}/v1/runs/${RUN_ID}/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "answers": [
      {"id": "TASK_ID_1", "answer": {"vendor_id": "x"}, "completion": []},
      {"id": "TASK_ID_2", "answer": {"decision": "HOLD", "evidence_set": []}, "completion": []},
      {"id": "TASK_ID_3", "answer": {"decision": "ESCALATE", "evidence_set": []}, "completion": []}
    ]
  }' | python3 -m json.tool
```

Replace `TASK_ID_1`, `TASK_ID_2`, and `TASK_ID_3` with ids from Step 1.

**What to expect in the submit response:**

- `"status": "complete"`
- `"per_task"` rows with `"score"` and `"breakdown"` containing `"clause": "§..."`
- `"by_tier"` stats for tiers 1, 2, and 3
- No `ground_truth`, `missing`, or `extra` fields

A deliberate stub with empty evidence often scores around 0.1. That is expected.

### Step 4: Fetch results

```bash
curl -s "${SANDBOX_URL}/v1/runs/${RUN_ID}" | python3 -m json.tool
```

This should match Step 3. Submitting the same payload again returns 200 (idempotent). A different payload after submit returns 409.

---

## Limits

| Limit | Value |
|---|---|
| Tasks per run | 1 to 10 |
| Rate limit | 30 requests per minute per IP |
| Request body size | 256 KB |
| Task pool | Fixed server-side (seed 9201). Callers cannot choose a seed |
| API keys | Not accepted. Submit answers only |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| First request very slow | Cold start on free tier. Retry `/health` after 60 seconds |
| `"Tool not available"` | Put tool `name` in the JSON body. Tier 1 cannot call `read_policy` |
| Submit 400 | Submit one answer per issued task id. Ids must match exactly |
| Submit 409 | Run already submitted with different answers. Create a new run |
| Judge score always 0 | Server may lack `GROQ_API_KEY`. Programmatic scores still apply |
