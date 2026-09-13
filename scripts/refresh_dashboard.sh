#!/usr/bin/env bash
# Rebuild dashboard/dist from committed results/ after eval updates.
# Does not call Groq — offline only.
#
# Usage:
#   ./scripts/refresh_dashboard.sh                  # uses dashboard/default_run_id.txt
#   ./scripts/refresh_dashboard.sh full-matrix-v2
#   WITH_BASELINES=1 ./scripts/refresh_dashboard.sh full-matrix-v2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-.ci-test-venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
fi

RUN_ID="${1:-$(cat dashboard/default_run_id.txt)}"
echo "run_id=${RUN_ID}"

# Refresh scores.json from transcripts when missing (common after resume/partial runs)
for run_dir in "${ROOT}"/results/groq-*/"${RUN_ID}"; do
  [[ -d "${run_dir}" ]] || continue
  if [[ -f "${run_dir}/transcript.jsonl" && ! -f "${run_dir}/scores.json" ]]; then
    echo "Recomputing scores: ${run_dir}"
    "${PYTHON}" -c "
from pathlib import Path
from evaluator_gym.eval.recompute import recompute_scores
recompute_scores(Path('${run_dir}'))
"
  fi
done

if [[ "${WITH_BASELINES:-0}" == "1" ]]; then
  echo "Computing adversarial baselines for ${RUN_ID}..."
  "${PYTHON}" scripts/compute_adversarial_baselines.py --run-id "${RUN_ID}" --all
fi

echo "${RUN_ID}" > dashboard/default_run_id.txt
"${PYTHON}" -m dashboard.build --run-id "${RUN_ID}"

echo ""
echo "Dashboard built: dashboard/dist/"
echo "  Local preview:  cd dashboard/dist && python -m http.server 8765"
echo "  Deploy update:  git add results/ dashboard/default_run_id.txt"
echo "                  git commit -m 'Update eval results and dashboard data'"
echo "                  git push origin main"
echo "  (GitHub Actions will rebuild and publish Pages automatically on push.)"
