from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from replaylab.analysis import run_analysis
from replaylab.state import AnalysisResult
from replaylab.trajectory import ExecutionSettings, ParsedTrajectory, TrajectoryAction


class AnalysisTests(unittest.TestCase):
    def test_shared_analysis_runs_pipeline_and_reports_progress(self):
        parsed = ParsedTrajectory(
            "mini-swe-agent-1.1",
            (TrajectoryAction(1, "pwd"),),
            ExecutionSettings(),
        )
        resolved = object()
        result = AnalysisResult("instance", "dataset", "test", "image", "base", "temporary", "start")
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "report.html"
            json_path = Path(directory) / "custom.json"
            with (
                patch("replaylab.analysis.load_trajectory", return_value=parsed) as load,
                patch("replaylab.analysis.resolve_instance", return_value=resolved) as resolve,
                patch("replaylab.analysis.replay_trajectory", return_value=result) as replay,
                patch("replaylab.analysis.write_reports", return_value=(html, json_path)) as write,
            ):
                artifacts = run_analysis(
                    "input.json",
                    "instance",
                    html,
                    dataset="dataset",
                    split="train",
                    timeout=19,
                    allow_network=True,
                    json_path=json_path,
                    trajectory_label="uploaded.traj.json",
                    progress=messages.append,
                )
        self.assertIs(artifacts.result, result)
        self.assertEqual(result.trajectory_path, "uploaded.traj.json")
        load.assert_called_once_with("input.json")
        resolve.assert_called_once_with("dataset", "train", "instance")
        self.assertEqual(replay.call_args.kwargs["timeout_override"], 19)
        self.assertTrue(replay.call_args.kwargs["allow_network"])
        write.assert_called_once_with(result, html, json_path)
        self.assertEqual(messages[0], "Loading trajectory")
        self.assertIn("Loaded 1 actions", messages)
        self.assertEqual(messages[-1], "Report ready")
