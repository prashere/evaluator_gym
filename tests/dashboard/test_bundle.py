from pathlib import Path

from dashboard.bundle import build_bundle
from dashboard.load import VersionTriple, load_run

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def test_bundle_v2_structure():
    run_a = load_run(FIXTURE / "run-a", slug="groq-gpt-oss-20b", is_baseline=False, on_duplicate="fail")
    run_b = load_run(FIXTURE / "run-b", slug="groq-gpt-oss-120b", is_baseline=False, on_duplicate="fail")
    version = VersionTriple("1.0.0", "0.1.1", "1.0.0")
    bundle = build_bundle(
        [run_a, run_b],
        version,
        {"seed-003": {"notes": "test", "coverage_tier": 1}},
        run_id="test-run",
        duplicate_warnings=[],
    )
    assert bundle["version"] == 2
    assert bundle["meta"]["subset_warning"] is True
    assert len(bundle["models"]) == 2
    assert bundle["models"][0]["diagnostics"]["strengths"] is not None
    task = next(t for t in bundle["tasks"] if t["task_id"] == "seed-003")
    assert task["task_type"] == "Retrieval"
    not_run = bundle["heatmap"]["cells"]["seed-001|groq-gpt-oss-20b"]
    assert not_run["bucket"] == "not_run"
    parse_cell = bundle["heatmap"]["cells"]["seed-003|groq-gpt-oss-120b"]
    assert parse_cell["bucket"] == "format_failure"
    assert len(bundle["inspectors"]) >= 4
    inspector = bundle["inspectors"]["seed-016|groq-gpt-oss-20b|0"]
    assert inspector["chain"]["recorded_audit"]["rubric_checks"]
