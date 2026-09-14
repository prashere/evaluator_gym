#!/usr/bin/env bash
# Reviewer-style round trip — mirrors docs/sandbox_manual_walkthrough.md Steps 2–7.
# Usage: SANDBOX_URL=https://your-service.onrender.com ./scripts/sandbox_reviewer_roundtrip.sh
set -euo pipefail

BASE_URL="${SANDBOX_URL:-http://localhost:8080}"

echo "==> Step 2: Create tier-2 run (n=1)"
RUN_JSON=$(curl -sf -X POST "${BASE_URL}/v1/runs" \
  -H "Content-Type: application/json" \
  -d '{"tier": "2", "n": 1}')
RUN_ID=$(echo "${RUN_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
TASK_ID=$(echo "${RUN_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['tasks'][0]['id'])")
echo "run_id=${RUN_ID} task_id=${TASK_ID}"

echo "==> Step 3: read_document case_context.json"
curl -sf -X POST "${BASE_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "read_document", "arguments": {"doc_id": "case_context.json"}}' \
  | python3 -c "import sys,json; b=json.load(sys.stdin); assert b.get('error') is None and b.get('result'); print('OK case_context')"

echo "==> Step 4: read_policy"
curl -sf -X POST "${BASE_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "read_policy", "arguments": {}}' \
  | python3 -c "import sys,json; b=json.load(sys.stdin); assert b.get('error') is None and 'Domain' in (b.get('result') or ''); print('OK policy')"

echo "==> Step 6: Submit patched answer (walkthrough stub)"
SUBMIT=$(curl -sf -X POST "${BASE_URL}/v1/runs/${RUN_ID}/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"answers\": [{
      \"id\": \"${TASK_ID}\",
      \"answer\": {\"decision\": \"HOLD\", \"evidence_set\": []},
      \"completion\": [{\"role\": \"tool\", \"content\": \"reviewer round trip\"}]
    }]
  }")
echo "${SUBMIT}" | python3 -c "
import sys, json
b = json.load(sys.stdin)
assert b.get('status') == 'complete', b
assert 'ground_truth' not in json.dumps(b)
row = b['per_task'][0]
assert row['scored'] and row['breakdown']
for k, v in row['breakdown'].items():
    assert v['clause'].startswith('§'), (k, v)
print('OK submit score=', row['score'])
"

echo "==> Step 7: GET run summary"
curl -sf "${BASE_URL}/v1/runs/${RUN_ID}" \
  | python3 -c "import sys,json; b=json.load(sys.stdin); assert b.get('status')=='complete'; print('OK cached results')"

echo "Reviewer round trip PASSED against ${BASE_URL}"
