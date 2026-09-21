from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence
from uuid import uuid4

from replaylab.trajectory import TrajectoryAction


class StateError(RuntimeError):
    pass


class CommandResult(Protocol):
    stdout: str
    returncode: int


class GitExecutor(Protocol):
    def exec_internal(
        self,
        argv: Sequence[str],
        *,
        cwd: str = "/testbed",
        env: dict[str, str] | None = None,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class RepoSnapshot:
    head: str
    worktree_tree: str
    index_tree: str
    status: str


@dataclass(frozen=True)
class FileChange:
    path: str
    status: str
    origins: tuple[str, ...]
    old_path: str | None = None


@dataclass
class StepResult:
    action: TrajectoryAction
    replay_stdout: str
    replay_returncode: int | None
    timed_out: bool
    duration_seconds: float
    output_match: bool | None
    before: RepoSnapshot
    after: RepoSnapshot | None
    changes: list[FileChange] = field(default_factory=list)
    worktree_diff: str = ""
    index_diff: str = ""
    signals: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    instance_id: str
    dataset: str
    split: str
    image: str
    base_commit: str
    trajectory_path: str
    started_at: str
    recorded_image: str | None = None
    image_digest: str | None = None
    image_source: str | None = None
    completed_at: str = ""
    baseline: RepoSnapshot | None = None
    steps: list[StepResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "replaylab-1", **asdict(self)}


class StateTracker:
    def __init__(self, executor: GitExecutor, repo_dir: str = "/testbed") -> None:
        self.executor = executor
        self.repo_dir = repo_dir

    def snapshot(self) -> RepoSnapshot:
        head = self._git("rev-parse", "HEAD").strip()
        if not head:
            raise StateError("git rev-parse HEAD returned no commit")
        index_tree = self._git("write-tree").strip()

        temporary_index = f"/tmp/replaylab-index-{uuid4().hex}"
        alternate_env = {"GIT_INDEX_FILE": temporary_index}
        try:
            self._git("read-tree", "HEAD", env=alternate_env)
            self._git("add", "-A", "--", env=alternate_env)
            worktree_tree = self._git("write-tree", env=alternate_env).strip()
        finally:
            self.executor.exec_internal(["rm", "-f", temporary_index], cwd=self.repo_dir)

        status = self._git("status", "--short", "--untracked-files=all")
        return RepoSnapshot(
            head=head,
            worktree_tree=worktree_tree,
            index_tree=index_tree,
            status=status,
        )

    def compare(
        self, before: RepoSnapshot, after: RepoSnapshot
    ) -> tuple[list[FileChange], str, str]:
        worktree_diff = self._tree_diff(before.worktree_tree, after.worktree_tree)
        index_diff = self._tree_diff(before.index_tree, after.index_tree)
        worktree_changes = self._name_status(
            before.worktree_tree, after.worktree_tree, "worktree"
        )
        index_changes = self._name_status(before.index_tree, after.index_tree, "index")
        return _merge_changes([*worktree_changes, *index_changes]), worktree_diff, index_diff

    def verify_base_commit(self, base_commit: str, snapshot: RepoSnapshot) -> list[str]:
        if not base_commit:
            raise StateError("Dataset row has no base_commit")
        self._git("cat-file", "-e", f"{base_commit}^{{commit}}")
        ancestor = self.executor.exec_internal(
            ["git", "merge-base", "--is-ancestor", base_commit, snapshot.head],
            cwd=self.repo_dir,
        )
        if ancestor.returncode != 0:
            raise StateError(
                f"Dataset base commit {base_commit} is not an ancestor of image HEAD {snapshot.head}"
            )
        if snapshot.head != base_commit:
            return [
                f"Image HEAD {snapshot.head} is above dataset base commit {base_commit}; preserving the prepared image state."
            ]
        return []

    def _tree_diff(self, old_tree: str, new_tree: str) -> str:
        if old_tree == new_tree:
            return ""
        return self._git(
            "diff",
            "--binary",
            "--find-renames",
            "--no-ext-diff",
            old_tree,
            new_tree,
            "--",
        )

    def _name_status(self, old_tree: str, new_tree: str, origin: str) -> list[FileChange]:
        if old_tree == new_tree:
            return []
        raw = self._git(
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            old_tree,
            new_tree,
            "--",
        )
        return _parse_name_status(raw, origin)

    def _git(self, *args: str, env: dict[str, str] | None = None) -> str:
        result = self.executor.exec_internal(
            ["git", *args], cwd=self.repo_dir, env=env
        )
        if result.returncode != 0:
            detail = result.stdout.strip()
            command = "git " + " ".join(args)
            raise StateError(f"{command} failed ({result.returncode}): {detail}")
        return result.stdout


def _parse_name_status(raw: str, origin: str) -> list[FileChange]:
    parts = raw.split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    changes: list[FileChange] = []
    index = 0
    while index < len(parts):
        code = parts[index]
        index += 1
        if not code or index >= len(parts):
            raise StateError("Malformed git --name-status output")
        kind = code[0]
        if kind in {"R", "C"}:
            if index + 1 >= len(parts):
                raise StateError("Malformed rename/copy in git --name-status output")
            old_path, path = parts[index], parts[index + 1]
            index += 2
        else:
            old_path, path = None, parts[index]
            index += 1
        status = {
            "A": "ADDED",
            "D": "DELETED",
            "R": "RENAMED",
            "C": "COPIED",
        }.get(kind, "MODIFIED")
        changes.append(FileChange(path=path, old_path=old_path, status=status, origins=(origin,)))
    return changes


def _merge_changes(changes: list[FileChange]) -> list[FileChange]:
    merged: dict[tuple[str, str, str | None], set[str]] = {}
    for change in changes:
        key = (change.path, change.status, change.old_path)
        merged.setdefault(key, set()).update(change.origins)
    return [
        FileChange(path=path, status=status, old_path=old_path, origins=tuple(sorted(origins)))
        for (path, status, old_path), origins in sorted(
            merged.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
        )
    ]
