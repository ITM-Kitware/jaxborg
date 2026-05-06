#!/usr/bin/env python
"""Run the dev transfer parity CLI."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.parity.bootstrap import configure_runtime

configure_runtime()

from scripts.dev.parity.transfer_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
