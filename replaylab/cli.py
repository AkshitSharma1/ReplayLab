from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from replaylab.analysis import DEFAULT_DATASET, run_analysis
from replaylab.docker_runner import ReplayError
from replaylab.trajectory import TrajectoryError


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
    ui = subcommands.add_parser("ui", help="Launch the local ReplayLab interface")
    ui.add_argument("--port", type=_port, default=8765)
    ui.add_argument("--output-dir", type=Path, default=Path("replaylab-output"))
    ui.add_argument("--no-open", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "ui":
        return _ui(args)
    return 2


def _analyze(args: argparse.Namespace) -> int:
    try:
        artifacts = run_analysis(
            args.trajectory,
            args.instance,
            args.output,
            dataset=args.dataset,
            split=args.split,
            timeout=args.timeout,
            allow_network=args.allow_network,
            json_path=args.json_output,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(f"HTML report: {artifacts.html_path}", file=sys.stderr)
        print(f"JSON report: {artifacts.json_path}", file=sys.stderr)
        if not artifacts.result.complete:
            print(f"Replay incomplete: {artifacts.result.error}", file=sys.stderr)
            return 1
        return 0
    except (TrajectoryError, ReplayError, ValueError, OSError) as exc:
        print(f"replaylab: error: {exc}", file=sys.stderr)
        return 2


def _ui(args: argparse.Namespace) -> int:
    from replaylab.ui import run_ui

    try:
        run_ui(port=args.port, output_dir=args.output_dir, open_browser=not args.no_open)
        return 0
    except OSError as exc:
        print(f"replaylab: error: {exc}", file=sys.stderr)
        return 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
