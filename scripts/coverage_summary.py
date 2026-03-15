"""Print hierarchical verification coverage summary (L1/L2/L3/L4)."""

# ruff: noqa: E402

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.catalog import print_coverage_summary

if __name__ == "__main__":
    print_coverage_summary()
