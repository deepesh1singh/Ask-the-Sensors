from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def _run(module: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    print(f"\n{'=' * 70}\n>>> Running: {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nERROR: {module} exited with status {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)

def main() -> None:
    _run("scripts.setup_data")
    _run("scripts.build_features")
    _run("scripts.train")
    _run("scripts.evaluate")

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("  Figures:   outputs/figures/")
    print("  Models:    outputs/models/")
    print("  Metrics:   outputs/logs/evaluation_results.json")
    print("=" * 70)

if __name__ == "__main__":
    main()