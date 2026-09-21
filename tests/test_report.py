from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replaylab.report import write_reports
from replaylab.state import AnalysisResult, FileChange, RepoSnapshot, StepResult
from replaylab.trajectory import TrajectoryAction


class ReportTests(unittest.TestCase):
    def test_writes_escaped_html_and_round_trippable_json(self):
        snapshot = RepoSnapshot("abc", "tree", "tree", "")
        action = TrajectoryAction(
            1,
            "echo '<script>alert(1)</script>'",
            "snowman: ☃\n",
            0,
        )
        step = StepResult(
            action=action,
            replay_stdout="snowman: ☃\n",
            replay_returncode=0,
            timed_out=False,
            duration_seconds=0.25,
            output_match=True,
            before=snapshot,
            after=snapshot,
            changes=[FileChange("tests/test_x.py", "MODIFIED", ("worktree",))],
            worktree_diff="--- a/tests/test_x.py\n+++ b/tests/test_x.py\n",
            signals=["TEST_FILE_MODIFIED"],
        )
        result = AnalysisResult(
            instance_id="example__repo-1",
            dataset="dataset",
            split="test",
            image="image",
            base_commit="abc",
            trajectory_path="input.json",
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            baseline=snapshot,
            steps=[step],
            summary={
                "steps_replayed": 1,
                "steps_total": 1,
                "normalized_output_matches": 1,
                "recorded_outputs_available": 1,
                "steps_with_repository_changes": 1,
                "commands_timed_out": 0,
            },
            complete=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            html = Path(directory) / "report.html"
            html_path, json_path = write_reports(result, html)
            rendered = html_path.read_text(encoding="utf-8")
            self.assertNotIn("<script>alert(1)</script>", rendered)
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
            self.assertIn("TEST_FILE_MODIFIED", rendered)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "replaylab-1")
            self.assertEqual(payload["steps"][0]["replay_stdout"], "snowman: ☃\n")

    def test_rejects_same_html_and_json_path(self):
        result = AnalysisResult("i", "d", "s", "img", "base", "traj", "start")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "must be different"):
                write_reports(result, path, path)

    def test_partial_report_and_binary_patch_text_render(self):
        snapshot = RepoSnapshot("abc", "tree", "tree", " M binary.dat\n")
        step = StepResult(
            action=TrajectoryAction(1, "rewrite binary.dat"),
            replay_stdout="",
            replay_returncode=0,
            timed_out=False,
            duration_seconds=0.1,
            output_match=None,
            before=snapshot,
            after=None,
            worktree_diff="GIT binary patch\nliteral 2\nhW\n",
            error="state tracker unavailable <unsafe>",
        )
        result = AnalysisResult(
            "i", "d", "s", "img", "base", "traj", "start",
            completed_at="end",
            baseline=snapshot,
            steps=[step],
            summary={
                "steps_replayed": 1,
                "steps_total": 2,
                "normalized_output_matches": 0,
                "recorded_outputs_available": 0,
                "steps_with_repository_changes": 0,
                "commands_timed_out": 0,
            },
            complete=False,
            error="stopped <early>",
        )
        with tempfile.TemporaryDirectory() as directory:
            html_path, _ = write_reports(result, Path(directory) / "partial.html")
            rendered = html_path.read_text(encoding="utf-8")
            self.assertIn("GIT binary patch", rendered)
            self.assertIn("stopped &lt;early&gt;", rendered)
            self.assertIn("state tracker unavailable &lt;unsafe&gt;", rendered)


if __name__ == "__main__":
    unittest.main()
