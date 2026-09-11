from fastapi.testclient import TestClient

from evaluator_gym.sandbox.app import app

FORBIDDEN_KEYS = {"ground_truth", "answer"}


def test_tasks_response_has_no_ground_truth():
    client = TestClient(app)
    resp = client.get("/tasks?tier=2&n=2&seed=7")
    assert resp.status_code == 200
    payload = resp.json()

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k not in FORBIDDEN_KEYS, f"leak at {path}.{k}"
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(payload)
