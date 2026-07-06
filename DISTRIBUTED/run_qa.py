#!/usr/bin/env python

from pathlib import Path
import runpy
import sys


def main() -> None:
    baseline_dir = Path(__file__).resolve().parents[1] / "BASELINE"
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))

    runpy.run_path(str(baseline_dir / "run_qa.py"), run_name="__main__")


if __name__ == "__main__":
    main()