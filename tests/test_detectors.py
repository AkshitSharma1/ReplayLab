from __future__ import annotations

import unittest

from replaylab.detectors import (
    TEST_CONFIG_MODIFIED,
    TEST_FILE_DELETED,
    TEST_FILE_MODIFIED,
    detect_signals,
    is_test_config,
    is_test_file,
)
from replaylab.state import FileChange


class DetectorTests(unittest.TestCase):
    def test_test_path_rules(self):
        self.assertTrue(is_test_file("tests/test_auth.py"))
        self.assertTrue(is_test_file("src/pkg/test/test_auth.py"))
        self.assertTrue(is_test_file("pkg/auth_test.py"))
        self.assertFalse(is_test_file("src/contest/parser.py"))

    def test_config_path_rules(self):
        self.assertTrue(is_test_config("pyproject.toml"))
        self.assertTrue(is_test_config("nested/tox.ini"))
        self.assertTrue(is_test_config(".github/workflows/tests.yml"))
        self.assertFalse(is_test_config(".github/dependabot.yml"))

    def test_all_signals_are_stable_and_deduplicated(self):
        changes = [
            FileChange("tests/test_auth.py", "MODIFIED", ("worktree",)),
            FileChange("tests/test_auth.py", "MODIFIED", ("index",)),
            FileChange("tests/old.py", "DELETED", ("worktree",)),
            FileChange("pyproject.toml", "MODIFIED", ("index",)),
        ]
        self.assertEqual(
            detect_signals(changes),
            [TEST_FILE_MODIFIED, TEST_FILE_DELETED, TEST_CONFIG_MODIFIED],
        )

    def test_rename_away_is_deletion(self):
        signals = detect_signals(
            [FileChange("src/test_auth.py", "RENAMED", ("worktree",), "tests/test_auth.py")]
        )
        self.assertEqual(signals, [TEST_FILE_MODIFIED, TEST_FILE_DELETED])

    def test_source_change_has_no_signal(self):
        self.assertEqual(
            detect_signals([FileChange("src/auth.py", "MODIFIED", ("worktree",))]), []
        )


if __name__ == "__main__":
    unittest.main()

