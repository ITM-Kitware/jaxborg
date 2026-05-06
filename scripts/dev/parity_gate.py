#!/usr/bin/env python
"""Run the dev parity merge gate."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.parity.gate import main

if __name__ == "__main__":
    raise SystemExit(main())
