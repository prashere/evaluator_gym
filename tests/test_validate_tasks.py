import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_seed_tasks_script_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_tasks.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
