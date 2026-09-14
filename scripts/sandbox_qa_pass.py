#!/usr/bin/env python3
"""One-shot senior QA pass against live sandbox — exit 0 if all checks pass."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("SANDBOX_URL", "http://localhost:8080").rstrip("/")


def req(method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        failures.append(line)


failures: list[str] = []

# 1. Health
code, body = req("GET", "/health")
check("GET /health", code == 200 and body.get("status") == "ok", str(body))

# 2. Legacy blocked
code, _ = req("GET", "/tasks")
check("GET /tasks → 404", code == 404)
code, _ = req("POST", "/submit", {})
check("POST /submit → 404", code == 404)

# 3. Balanced tier=all n=3
code, run = req("POST", "/v1/runs", {"tier": "all", "n": 3})
check("POST /v1/runs tier=all n=3", code == 200, f"tier_mix={run.get('tier_mix')}")
mix = run.get("tier_mix") or {}
check("tier_mix 1+1+1", mix == {"tier_1": 1, "tier_2": 1, "tier_3": 1}, str(mix))
run_id = run.get("run_id", "")
tasks = run.get("tasks") or []
check("3 tasks issued", len(tasks) == 3)
tiers = sorted(t["tier"] for t in tasks)
check("tiers 1,2,3 present", tiers == [1, 2, 3])

# 4. No GT leak on create
text = json.dumps(run)
check("no ground_truth on create", "ground_truth" not in text)

# 5. Tools per tier
t2 = next(t for t in tasks if t["tier"] == 2)
t1 = next(t for t in tasks if t["tier"] == 1)
tid2, tid1 = t2["id"], t1["id"]
check("tier2 has read_policy", "read_policy" in t2.get("tools_available", []))
check("tier1 no read_policy", "read_policy" not in t1.get("tools_available", []))

code, tool = req(
    "POST",
    f"/v1/runs/{run_id}/tasks/{tid2}/tools",
    {"name": "read_policy", "arguments": {}},
)
check("read_policy tier2", code == 200 and tool.get("error") is None and "Domain" in (tool.get("result") or ""))

code, tool = req(
    "POST",
    f"/v1/runs/{run_id}/tasks/{tid1}/tools",
    {"name": "read_policy", "arguments": {}},
)
check("read_policy tier1 blocked", tool.get("error") is not None)

code, tool = req(
    "POST",
    f"/v1/runs/{run_id}/tasks/{tid2}/tools",
    {"name": "read_document", "arguments": {"doc_id": "fake.json"}},
)
check("bad doc generic error", tool.get("error", {}).get("code") == "TOOL_ERROR")

# 6. Scoring — tier 2 stub vs schema fail
code, stub = req(
    "POST",
    f"/v1/runs/{run_id}/submit",
    {
        "answers": [
            {
                "id": t["id"],
                "answer": (
                    {"vendor_id": "x"}
                    if t["tier"] == 1
                    else {"decision": "HOLD", "evidence_set": []}
                ),
                "completion": [{"role": "tool", "content": "qa"}] if t["tier"] >= 2 else [],
            }
            for t in tasks
        ]
    },
)
check("submit all 3 tasks", code == 200 and stub.get("status") == "complete")
check("by_tier populated", stub.get("by_tier", {}).get("tier_2", {}).get("n") == 1)
per = {r["id"]: r for r in stub.get("per_task", [])}
check("all tasks scored row", len(per) == 3)
for row in per.values():
    if row.get("scored"):
        check(
            f"breakdown has clauses {row['id']}",
            all("clause" in v and v["clause"].startswith("§") for v in (row.get("breakdown") or {}).values()),
        )
check("no ground_truth on submit", "ground_truth" not in json.dumps(stub))
check("no missing/extra keys", "missing" not in json.dumps(stub) and "extra" not in json.dumps(stub))

# 7. Idempotency + 409
code, again = req(
    "POST",
    f"/v1/runs/{run_id}/submit",
    {
        "answers": [
            {
                "id": t["id"],
                "answer": (
                    {"vendor_id": "x"}
                    if t["tier"] == 1
                    else {"decision": "HOLD", "evidence_set": []}
                ),
                "completion": [{"role": "tool", "content": "qa"}] if t["tier"] >= 2 else [],
            }
            for t in tasks
        ]
    },
)
check("idempotent resubmit", code == 200 and again == stub)

code, conflict = req(
    "POST",
    f"/v1/runs/{run_id}/submit",
    {
        "answers": [
            {
                "id": t["id"],
                "answer": (
                    {"vendor_id": "y"}
                    if t["tier"] == 1
                    else {"decision": "APPROVE", "evidence_set": []}
                ),
                "completion": [{"role": "tool", "content": "changed"}] if t["tier"] >= 2 else [],
            }
            for t in tasks
        ]
    },
)
check("409 on changed submit", code == 409)

code, got = req("GET", f"/v1/runs/{run_id}")
check("GET matches submit", code == 200 and got.get("mean_reward") == stub.get("mean_reward"))

# 8. Partial submit rejected (new run)
code, run2 = req("POST", "/v1/runs", {"tier": "2", "n": 2})
run2_id = run2["run_id"]
code, partial = req(
    "POST",
    f"/v1/runs/{run2_id}/submit",
    {
        "answers": [
            {
                "id": run2["tasks"][0]["id"],
                "answer": {"decision": "HOLD", "evidence_set": []},
                "completion": [{"role": "tool", "content": "x"}],
            }
        ],
    },
)
check("partial submit 400", code == 400)

print()
if failures:
    print(f"QA VERDICT: BLOCKED — {len(failures)} failure(s)")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("QA VERDICT: PASS — all live checks OK")
sys.exit(0)
