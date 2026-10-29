"""Collate a retained record file using an explicit selection file."""

import argparse
import json
from pathlib import Path

from . import collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collate(json.loads(args.records.read_text()), **json.loads(args.selection.read_text()))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
