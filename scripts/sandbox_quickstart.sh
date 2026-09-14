#!/usr/bin/env bash
# Local sandbox walkthrough — v1 run API (tool mode).
# Prerequisite: server running — python -m evaluator_gym.sandbox
set -euo pipefail

BASE_URL="${SANDBOX_URL:-http://localhost:8080}"

echo "==> Health"
curl -sf "${BASE_URL}/health" | python3 -m json.tool

echo "==> Create run (tier 2, n=1)"
RUN_JSON=$(curl -sf -X POST "${BASE_URL}/v1/runs" \
  -H "Content-Type: application/json" \
  -d '{"tier": "2", "n": 1}')
echo "${RUN_JSON}" | python3 -m json.tool

RUN_ID=$(echo "${RUN_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
TASK_ID=$(echo "${RUN_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['tasks'][0]['id'])")

echo "==> Tool call: read case_context.json"
curl -sf -X POST "${BASE_URL}/v1/runs/${RUN_ID}/tasks/${TASK_ID}/tools" \
  -H "Content-Type: application/json" \
  -d '{"name": "read_document", "arguments": {"doc_id": "case_context.json"}}' \
  | python3 -m json.tool

echo "==> Submit stub answer (expect low score / parse or judge failure)"
curl -sf -X POST "${BASE_URL}/v1/runs/${RUN_ID}/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"answers\": [{
      \"id\": \"${TASK_ID}\",
      \"answer\": {\"decision\": \"HOLD\", \"evidence_set\": []},
      \"completion\": [{\"role\": \"tool\", \"content\": \"stub transcript\"}]
    }]
  }" | python3 -m json.tool

echo "==> Get run summary"
curl -sf "${BASE_URL}/v1/runs/${RUN_ID}" | python3 -m json.tool

echo "Done. Run id: ${RUN_ID}"
