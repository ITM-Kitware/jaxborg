"""Index an already copied set of records for the completed MAPPO response study."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze import resolve_config

SOURCE = "af6bc98321dd4e9e892d76b0369e0d8e"
REPORT = "fd73b4b6b6ed400aab12dc6a96dfe56b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="New bundle containing producer/<run>/artifacts/")
    parser.add_argument(
        "--source-store", required=True, help="Original tracking-store identity, retained as provenance"
    )
    args = parser.parse_args()
    root = args.input.resolve()
    if (root / "index.json").exists() or (root / "analysis.json").exists():
        parser.error("Bundle is already indexed; prepare a new input directory")
    protocol = json.loads((root / "producer" / SOURCE / "artifacts/protocol/manifest.json").read_text())
    configuration = resolve_config(protocol, {}, {})
    (root / "analysis.json").write_text(json.dumps(configuration, indent=2) + "\n")
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        reference = None
        if relative.parts[0] == "producer":
            if len(relative.parts) < 4 or relative.parts[2] != "artifacts":
                parser.error(f"Unexpected producer file layout: {relative}")
            reference = f"runs:/{relative.parts[1]}/{'/'.join(relative.parts[3:])}"
        files[relative.as_posix()] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
            "reference": reference,
        }
    index = {
        "format": "frozen-defender-response-v1",
        "source_store": args.source_store,
        "protocol": f"runs:/{SOURCE}/protocol/manifest.json",
        "state": f"runs:/{REPORT}/results/state.json",
        "reference_summary": f"runs:/{REPORT}/results/summary.json",
        "configuration": "analysis.json",
        "files": files,
    }
    (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Indexed {len(files)} files in {root}")


if __name__ == "__main__":
    main()
