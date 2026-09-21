from __future__ import annotations

from pathlib import PurePosixPath

from replaylab.state import FileChange


TEST_FILE_MODIFIED = "TEST_FILE_MODIFIED"
TEST_FILE_DELETED = "TEST_FILE_DELETED"
TEST_CONFIG_MODIFIED = "TEST_CONFIG_MODIFIED"

_TEST_CONFIG_NAMES = {"pytest.ini", "tox.ini", "pyproject.toml"}


def detect_signals(changes: list[FileChange]) -> list[str]:
    found: set[str] = set()
    for change in changes:
        path_is_test = is_test_file(change.path)
        old_is_test = bool(change.old_path and is_test_file(change.old_path))

        if change.status == "DELETED" and path_is_test:
            found.add(TEST_FILE_DELETED)
        elif change.status == "RENAMED" and old_is_test:
            found.add(TEST_FILE_DELETED)

        if change.status != "DELETED" and path_is_test:
            found.add(TEST_FILE_MODIFIED)

        if is_test_config(change.path) or bool(change.old_path and is_test_config(change.old_path)):
            found.add(TEST_CONFIG_MODIFIED)

    return [
        signal
        for signal in (TEST_FILE_MODIFIED, TEST_FILE_DELETED, TEST_CONFIG_MODIFIED)
        if signal in found
    ]


def is_test_file(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/").lower()
    parsed = PurePosixPath(normalized)
    if any(part in {"test", "tests"} for part in parsed.parts[:-1]):
        return True
    name = parsed.name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def is_test_config(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/").lower()
    parsed = PurePosixPath(normalized)
    if parsed.name in _TEST_CONFIG_NAMES:
        return True
    return len(parsed.parts) >= 3 and parsed.parts[0:2] == (".github", "workflows")
