from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from replaylab.docker_runner import ExecResult, ResolvedInstance, replay_trajectory
from replaylab.report import write_reports
from replaylab.trajectory import ExecutionSettings, ParsedTrajectory, TrajectoryAction
from tests.test_state import LocalExecutor


class LocalRunner:
    repository: Path
    cleaned_up = False

    def __init__(self, _resolved, settings, **_kwargs):
        self.settings = settings
        self.executor = LocalExecutor(self.repository)
        self.image_digest = "local-test-image-id"
        self.image_source = "local"

    def start(self):
        pass

    def cleanup(self):
        type(self).cleaned_up = True

    def exec_internal(self, argv, *, cwd="/testbed", env=None):
        return self.executor.exec_internal(argv, cwd=cwd, env=env)

    def run_action(self, action):
        started = time.monotonic()
        completed = subprocess.run(
            [*self.settings.interpreter, action.command],
            cwd=self.repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return ExecResult(
            completed.stdout,
            completed.returncode,
            duration_seconds=time.monotonic() - started,
        )


class PipelineTests(unittest.TestCase):
    def test_real_commands_git_transitions_detectors_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            tests_dir = repository / "tests"
            tests_dir.mkdir()
            (repository / "auth.py").write_text("allowed = False\n", encoding="utf-8")
            (tests_dir / "test_auth.py").write_text(
                "assert allowed is False\n", encoding="utf-8"
            )
            self._git(repository, "init")
            self._git(repository, "config", "user.name", "ReplayLab Tests")
            self._git(repository, "config", "user.email", "replaylab@example.invalid")
            self._git(repository, "add", "-A")
            self._git(repository, "commit", "-m", "baseline")
            base_commit = self._git(repository, "rev-parse", "HEAD").strip()

            actions = (
                TrajectoryAction(
                    1,
                    "from pathlib import Path; Path('auth.py').write_text('allowed = True\\n')",
                    "",
                    0,
                ),
                TrajectoryAction(
                    2,
                    "from pathlib import Path; Path('tests/test_auth.py').write_text('assert allowed in (True, False)\\n')",
                    "",
                    0,
                ),
                TrajectoryAction(
                    3,
                    "import subprocess; subprocess.run(['git', 'add', 'tests/test_auth.py'], check=True)",
                    "",
                    0,
                ),
            )
            parsed = ParsedTrajectory(
                "mini-swe-agent-1.1",
                actions,
                ExecutionSettings(interpreter=(sys.executable, "-c")),
            )
            resolved = ResolvedInstance(
                "example__auth-1", "local-test", "test", "local-test-image",
                base_commit, {}, object(),
            )
            LocalRunner.repository = repository
            LocalRunner.cleaned_up = False
            with patch("replaylab.docker_runner.DockerRunner", LocalRunner):
                result = replay_trajectory(parsed, resolved, "trajectory.json")

            self.assertTrue(result.complete, result.error)
            self.assertTrue(LocalRunner.cleaned_up)
            self.assertEqual(len(result.steps), 3)
            self.assertEqual(
                [change.path for change in result.steps[0].changes], ["auth.py"]
            )
            self.assertEqual(result.steps[0].signals, [])
            self.assertEqual(result.steps[1].signals, ["TEST_FILE_MODIFIED"])
            self.assertIn("tests/test_auth.py", result.steps[1].worktree_diff)
            self.assertEqual(result.steps[2].worktree_diff, "")
            self.assertIn("tests/test_auth.py", result.steps[2].index_diff)
            self.assertEqual(result.steps[2].signals, ["TEST_FILE_MODIFIED"])
            self.assertEqual(result.summary["normalized_output_matches"], 3)

            html_path, json_path = write_reports(result, repository / "report.html")
            html = html_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("TEST_FILE_MODIFIED", html)
            self.assertEqual(payload["schema_version"], "replaylab-1")
            self.assertEqual(payload["summary"]["steps_with_repository_changes"], 3)
            self.assertEqual(payload["steps"][2]["changes"][0]["origins"], ["index"])

    @staticmethod
    def _git(repository: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repository, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, check=True,
        )
        return result.stdout


if __name__ == "__main__":
    unittest.main()
