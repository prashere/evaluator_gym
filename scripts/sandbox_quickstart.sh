#!/usr/bin/env bash
# Two-minute reviewer quickstart — fetch tasks, submit stub answer, print score.
set -euo pipefail

BASE_URL="${SANDBOX_URL:-http://localhost:8080}"

echo "==> Health"
curl -sf "${BASE_URL}/health" | python3 -m json.tool

echo "==> Fetch tasks"
TASKS_JSON=$(curl -sf "${BASE_URL}/tasks?tier=2&n=2&seed=7")
echo "${TASKS_JSON}" | python3 -m json.tool

RUN_ID=$(echo "${TASKS_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
TASK_ID=$(echo "${TASKS_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['tasks'][0]['id'])")

echo "==> Submit answer for run_id=${RUN_ID}"
curl -sf -X POST "${BASE_URL}/submit" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"${RUN_ID}\", \"answers\": [{\"id\": \"${TASK_ID}\", \"answer\": {\"status\": \"stub\"}}]}" \
  | python3 -m json.tool

echo "Done."
