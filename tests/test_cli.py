from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replaylab.cli import main
from replaylab.docker_runner import ResolvedInstance
from replaylab.state import AnalysisResult


def write_trajectory(path: Path) -> None:
    path.write_text(json.dumps({
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {"config": {"environment": {}}},
        "messages": [{
            "role": "assistant",
            "extra": {"actions": [{"command": "pwd"}]},
        }],
    }), encoding="utf-8")


class CliTests(unittest.TestCase):
    def test_one_command_writes_html_and_default_json_sidecar(self):
        resolved = ResolvedInstance(
            "example__repo-1", "dataset", "test", "image", "base", {}, object()
        )
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.traj.json"
            write_trajectory(trajectory)
            html = Path(directory) / "report.html"
            result = AnalysisResult(
                "example__repo-1", "dataset", "test", "image", "base", str(trajectory), "start",
                completed_at="end",
                complete=True,
                summary={
                    "steps_total": 2,
                    "steps_replayed": 2,
                    "normalized_output_matches": 0,
                    "recorded_outputs_available": 0,
                    "steps_with_repository_changes": 0,
                    "commands_timed_out": 0,
                },
            )
            with (
                patch("replaylab.cli.resolve_instance", return_value=resolved),
                patch("replaylab.cli.replay_trajectory", return_value=result) as replay,
            ):
                exit_code = main([
                    "analyze", "--trajectory", str(trajectory), "--instance", "example__repo-1",
                    "--output", str(html), "--dataset", "dataset", "--timeout", "17",
                    "--allow-network",
                ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(html.exists())
            payload = json.loads(html.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "replaylab-1")
            replay.assert_called_once()
            self.assertTrue(replay.call_args.kwargs["allow_network"])
            self.assertEqual(replay.call_args.kwargs["timeout_override"], 17)

    def test_partial_replay_still_writes_requested_json_path(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.traj.json"
            write_trajectory(trajectory)
            result = AnalysisResult(
                "example__repo-1", "dataset", "test", "image", "base", str(trajectory), "start",
                completed_at="end", complete=False, error="container failed",
                summary={
                    "steps_total": 2,
                    "steps_replayed": 0,
                    "normalized_output_matches": 0,
                    "recorded_outputs_available": 0,
                    "steps_with_repository_changes": 0,
                    "commands_timed_out": 0,
                },
            )
            html = Path(directory) / "partial.html"
            json_path = Path(directory) / "custom.json"
            with (
                patch("replaylab.cli.resolve_instance", return_value=object()),
                patch("replaylab.cli.replay_trajectory", return_value=result),
            ):
                exit_code = main([
                    "analyze", "--trajectory", str(trajectory), "--instance", "example__repo-1",
                    "--output", str(html), "--json-output", str(json_path),
                ])
            self.assertEqual(exit_code, 1)
            self.assertTrue(html.exists())
            self.assertTrue(json_path.exists())
            self.assertFalse(html.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
