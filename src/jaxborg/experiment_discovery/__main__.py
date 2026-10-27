"""JSON command line boundary for retained evidence discovery."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _parser():
    parser = _ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--kind", choices=("frozen-defender", "evaluation-capture"), required=True)
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--selection")
    prepare.add_argument("--output", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--package", required=True)
    search = commands.add_parser("search")
    search.add_argument("--study", required=True)
    for key in ("protocol", "trial", "record"):
        search.add_argument("--" + key)
    inspect = commands.add_parser("inspect")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--output", required=True)
    for command in (publish, search):
        command.add_argument("--experiment", required=True)
    for command in (publish, search, inspect, fetch):
        command.add_argument("--tracking-uri", required=True)
    for command in (inspect, fetch):
        command.add_argument("--run-id", required=True)
        command.add_argument("--sha256", required=True)
    return parser


def _execute(args):
    if args.command == "prepare":
        from .package import package_paths, prepare_package

        manifest = prepare_package(args.kind, args.input, args.output, selection=args.selection)
        return {
            "sha256": manifest["sha256"],
            "kind": manifest["kind"],
            "package": str(Path(args.output).absolute()),
            "paths": package_paths(args.output),
        }
    from .mlflow import fetch_package, inspect_publication, publish_package, search_packages

    if args.command == "publish":
        return publish_package(args.package, args.tracking_uri, args.experiment)
    if args.command == "search":
        return search_packages(
            args.tracking_uri,
            args.experiment,
            study=args.study,
            protocol=args.protocol,
            trial=args.trial,
            record=args.record,
        )
    if args.command == "inspect":
        return inspect_publication(args.tracking_uri, args.run_id, args.sha256)
    return fetch_package(args.tracking_uri, args.run_id, args.sha256, args.output)


def main(argv=None):
    try:
        args = _parser().parse_args(argv)
        with redirect_stdout(sys.stderr):
            result = _execute(args)
    except Exception as error:
        details = {"type": type(error).__name__, "message": str(error)}
        if hasattr(error, "candidate_run_ids"):
            details["candidate_run_ids"] = error.candidate_run_ids
        print(json.dumps({"error": details}, sort_keys=True))
        return 1
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
