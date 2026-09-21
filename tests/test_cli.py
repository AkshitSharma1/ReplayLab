from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replaylab.analysis import AnalysisArtifacts
from replaylab.cli import main
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
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.traj.json"
            write_trajectory(trajectory)
            html = Path(directory) / "report.html"
            json_path = html.with_suffix(".json")
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
            artifacts = AnalysisArtifacts(result, html, json_path)
            with (
                patch("replaylab.cli.run_analysis", return_value=artifacts) as analyze,
            ):
                exit_code = main([
                    "analyze", "--trajectory", str(trajectory), "--instance", "example__repo-1",
                    "--output", str(html), "--dataset", "dataset", "--timeout", "17",
                    "--allow-network",
                ])
            self.assertEqual(exit_code, 0)
            analyze.assert_called_once()
            self.assertTrue(analyze.call_args.kwargs["allow_network"])
            self.assertEqual(analyze.call_args.kwargs["timeout"], 17)
            self.assertEqual(analyze.call_args.kwargs["dataset"], "dataset")

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
            artifacts = AnalysisArtifacts(result, html, json_path)
            with (
                patch("replaylab.cli.run_analysis", return_value=artifacts),
            ):
                exit_code = main([
                    "analyze", "--trajectory", str(trajectory), "--instance", "example__repo-1",
                    "--output", str(html), "--json-output", str(json_path),
                ])
            self.assertEqual(exit_code, 1)

    def test_ui_command_forwards_local_server_options(self):
        with patch("replaylab.ui.run_ui") as run_ui:
            exit_code = main([
                "ui", "--port", "9123", "--output-dir", "custom-output", "--no-open",
            ])
        self.assertEqual(exit_code, 0)
        run_ui.assert_called_once_with(
            port=9123,
            output_dir=Path("custom-output"),
            open_browser=False,
        )

    def test_ui_rejects_invalid_port(self):
        with self.assertRaises(SystemExit):
            main(["ui", "--port", "70000"])


if __name__ == "__main__":
    unittest.main()
