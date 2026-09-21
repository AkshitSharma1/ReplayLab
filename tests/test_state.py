from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from replaylab.detectors import TEST_CONFIG_MODIFIED, TEST_FILE_DELETED, TEST_FILE_MODIFIED, detect_signals
from replaylab.docker_runner import ExecResult
from replaylab.state import StateTracker


class LocalExecutor:
    def __init__(self, root: Path):
        self.root = root

    def exec_internal(
        self,
        argv: Sequence[str],
        *,
        cwd: str = "/testbed",
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        run_env = os.environ.copy()
        run_env.update(env or {})
        temporary_index = run_env.get("GIT_INDEX_FILE")
        if temporary_index and temporary_index.startswith("/tmp/"):
            run_env["GIT_INDEX_FILE"] = str(
                Path(tempfile.gettempdir()) / Path(temporary_index).name
            )
        if list(argv[:2]) == ["rm", "-f"]:
            target = Path(tempfile.gettempdir()) / Path(argv[2]).name
            target.unlink(missing_ok=True)
            target.with_suffix(target.suffix + ".lock").unlink(missing_ok=True)
            return ExecResult("", 0)
        result = subprocess.run(
            list(argv),
            cwd=self.root if cwd == "/testbed" else cwd,
            env=run_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return ExecResult(result.stdout, result.returncode)


class StateTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self._git("init")
        self._git("config", "user.name", "ReplayLab Tests")
        self._git("config", "user.email", "replaylab@example.invalid")
        (self.root / "src.py").write_text("value = 1\n", encoding="utf-8")
        (self.root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_old.py").write_text("def test_old(): pass\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-m", "baseline")
        self.tracker = StateTracker(LocalExecutor(self.root))

    def tearDown(self):
        self.temporary.cleanup()

    def _git(self, *args):
        subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        )

    def _git_output(self, *args):
        result = subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        )
        return result.stdout

    def test_tracks_worktree_add_modify_and_delete(self):
        before = self.tracker.snapshot()
        (self.root / "src.py").write_text("value = 2\n", encoding="utf-8")
        (self.root / "tests" / "test_new.py").write_text("def test_new(): pass\n", encoding="utf-8")
        (self.root / "tests" / "test_old.py").unlink()
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        self.assertIn("src.py", {change.path for change in changes})
        self.assertIn("tests/test_new.py", {change.path for change in changes})
        self.assertIn("tests/test_old.py", {change.path for change in changes})
        self.assertIn(TEST_FILE_MODIFIED, detect_signals(changes))
        self.assertIn(TEST_FILE_DELETED, detect_signals(changes))
        self.assertIn("+value = 2", worktree_diff)
        self.assertEqual(index_diff, "")

    def test_staging_dirty_baseline_is_an_index_change(self):
        (self.root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
        before = self.tracker.snapshot()
        self._git("add", "pyproject.toml")
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        self.assertEqual(worktree_diff, "")
        self.assertIn("pyproject.toml", index_diff)
        config_change = next(change for change in changes if change.path == "pyproject.toml")
        self.assertEqual(config_change.origins, ("index",))
        self.assertEqual(detect_signals(changes), [TEST_CONFIG_MODIFIED])

    def test_tracks_rename_in_worktree(self):
        before = self.tracker.snapshot()
        (self.root / "tests" / "test_old.py").rename(
            self.root / "tests" / "test_renamed.py"
        )
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        rename = next(change for change in changes if change.status == "RENAMED")
        self.assertEqual(rename.old_path, "tests/test_old.py")
        self.assertEqual(rename.path, "tests/test_renamed.py")
        self.assertEqual(rename.origins, ("worktree",))
        self.assertIn("rename from tests/test_old.py", worktree_diff)
        self.assertEqual(index_diff, "")
        self.assertEqual(
            detect_signals(changes), [TEST_FILE_MODIFIED, TEST_FILE_DELETED]
        )

    def test_untracked_binary_file_has_binary_safe_incremental_diff(self):
        before = self.tracker.snapshot()
        (self.root / "payload.bin").write_bytes(b"\x00\xff\x10\x00")
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        self.assertEqual([change.path for change in changes], ["payload.bin"])
        self.assertEqual(changes[0].status, "ADDED")
        self.assertIn("GIT binary patch", worktree_diff)
        self.assertEqual(index_diff, "")

    def test_base_commit_verification_and_ancestor_warning(self):
        base = self._git_output("rev-parse", "HEAD").strip()
        self.assertEqual(self.tracker.verify_base_commit(base, self.tracker.snapshot()), [])
        (self.root / "src.py").write_text("value = 99\n", encoding="utf-8")
        self._git("add", "src.py")
        self._git("commit", "-m", "later")
        warnings = self.tracker.verify_base_commit(base, self.tracker.snapshot())
        self.assertEqual(len(warnings), 1)
        self.assertIn("preserving", warnings[0])

    def test_clean_step_has_no_changes(self):
        before = self.tracker.snapshot()
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        self.assertEqual(changes, [])
        self.assertEqual(worktree_diff, "")
        self.assertEqual(index_diff, "")

    def test_commit_changes_head_but_preserves_content_snapshot(self):
        (self.root / "src.py").write_text("value = 3\n", encoding="utf-8")
        self._git("add", "src.py")
        before = self.tracker.snapshot()
        self._git("commit", "-m", "change")
        after = self.tracker.snapshot()
        changes, worktree_diff, index_diff = self.tracker.compare(before, after)
        self.assertNotEqual(before.head, after.head)
        self.assertEqual(changes, [])
        self.assertEqual(worktree_diff, "")
        self.assertEqual(index_diff, "")


if __name__ == "__main__":
    unittest.main()
