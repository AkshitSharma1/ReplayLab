from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from replaylab.docker_runner import ReplayError, replay_trajectory, resolve_instance
from replaylab.report import write_reports
from replaylab.trajectory import TrajectoryError, load_trajectory


DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replaylab",
        description="Replay coding-agent shell actions and report repository changes.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    analyze = subcommands.add_parser("analyze", help="Replay and analyze one trajectory")
    analyze.add_argument("--trajectory", type=Path, required=True)
    analyze.add_argument("--instance", required=True)
    analyze.add_argument("--output", type=Path, required=True, help="HTML report path")
    analyze.add_argument("--dataset", default=DEFAULT_DATASET)
    analyze.add_argument("--split", default="test")
    analyze.add_argument("--timeout", type=_positive_int)
    analyze.add_argument("--allow-network", action="store_true")
    analyze.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "analyze":
        return 2
    return _analyze(args)


def _analyze(args: argparse.Namespace) -> int:
    try:
        parsed = load_trajectory(args.trajectory)
        print(
            f"Loaded {len(parsed.actions)} actions from {args.trajectory}",
            file=sys.stderr,
        )
        print(f"Resolving {args.instance} in {args.dataset}/{args.split}", file=sys.stderr)
        resolved = resolve_instance(args.dataset, args.split, args.instance)
        result = replay_trajectory(
            parsed,
            resolved,
            args.trajectory,
            allow_network=args.allow_network,
            timeout_override=args.timeout,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        html_path, json_path = write_reports(result, args.output, args.json_output)
        print(f"HTML report: {html_path}", file=sys.stderr)
        print(f"JSON report: {json_path}", file=sys.stderr)
        if not result.complete:
            print(f"Replay incomplete: {result.error}", file=sys.stderr)
            return 1
        return 0
    except (TrajectoryError, ReplayError, ValueError, OSError) as exc:
        print(f"replaylab: error: {exc}", file=sys.stderr)
        return 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
