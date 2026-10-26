"""Import explicitly named local evidence collections into common records."""

import argparse
import json
from pathlib import Path

from jaxborg.experiment_records import validate_records

from .local import frozen_bundle, smoke_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-bundle", type=Path, required=True)
    parser.add_argument("--smoke-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("Generated evidence must be outside the source checkout")
    if any(output.is_relative_to(root.resolve()) for root in (args.frozen_bundle, args.smoke_store)):
        parser.error("Generated evidence must be separate from original evidence")
    records = frozen_bundle(args.frozen_bundle) + smoke_store(args.smoke_store)
    validate_records(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(records, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
