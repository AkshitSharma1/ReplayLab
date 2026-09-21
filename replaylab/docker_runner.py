from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from replaylab.detectors import detect_signals
from replaylab.state import AnalysisResult, StateError, StateTracker, StepResult
from replaylab.trajectory import ExecutionSettings, ParsedTrajectory, TrajectoryAction


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedInstance:
    instance_id: str
    dataset: str
    split: str
    image: str
    base_commit: str
    dataset_row: dict[str, Any]
    test_spec: Any


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    returncode: int
    timed_out: bool = False
    duration_seconds: float = 0.0


def resolve_instance(dataset: str, split: str, instance_id: str) -> ResolvedInstance:
    try:
        from swebench.harness.utils import load_swebench_dataset, make_test_spec
    except ImportError as exc:
        raise ReplayError(
            "SWE-bench 5.0.x is required. Install ReplayLab under Python 3.10-3.12."
        ) from exc

    try:
        rows = load_swebench_dataset(dataset, split, [instance_id])
    except Exception as exc:
        raise ReplayError(
            f"Could not load {instance_id!r} from dataset {dataset!r} split {split!r}: {exc}"
        ) from exc
    if len(rows) != 1:
        raise ReplayError(
            f"Expected one dataset row for {instance_id!r}, received {len(rows)}"
        )

    row = dict(rows[0])
    if row.get("instance_id") != instance_id:
        raise ReplayError(
            f"Dataset returned instance {row.get('instance_id')!r} instead of {instance_id!r}"
        )
    try:
        test_spec = make_test_spec(row)
    except Exception as exc:
        raise ReplayError(f"Could not create the SWE-bench test spec: {exc}") from exc

    image = getattr(test_spec, "image", None) or row.get("image")
    if not image:
        image = getattr(test_spec, "instance_image_key", None)
    if not isinstance(image, str) or not image:
        raise ReplayError("The SWE-bench test spec did not provide an instance image")

    base_commit = row.get("base_commit")
    if not isinstance(base_commit, str) or not base_commit:
        raise ReplayError("The SWE-bench dataset row did not provide base_commit")
    return ResolvedInstance(
        instance_id=instance_id,
        dataset=dataset,
        split=split,
        image=image,
        base_commit=base_commit,
        dataset_row=row,
        test_spec=test_spec,
    )


class DockerRunner:
    def __init__(
        self,
        resolved: ResolvedInstance,
        settings: ExecutionSettings,
        *,
        allow_network: bool = False,
        timeout_override: int | None = None,
        client: Any = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.resolved = resolved
        self.settings = settings
        self.allow_network = allow_network
        self.timeout = timeout_override or settings.timeout
        self.client = client
        self.container: Any = None
        self.image_digest: str | None = None
        self.image_source: str | None = None
        self.progress = progress or (lambda _message: None)

    def start(self) -> None:
        if self.client is None:
            try:
                import docker
            except ImportError as exc:
                raise ReplayError("Docker SDK is not installed") from exc
            try:
                self.client = docker.from_env(timeout=max(self.timeout + 30, 1800))
                self.client.ping()
            except Exception as exc:
                raise ReplayError(f"Docker is not available: {exc}") from exc

        self._ensure_image()
        try:
            image = self.client.images.get(self.resolved.image)
            digests = getattr(image, "attrs", {}).get("RepoDigests", [])
            self.image_digest = digests[0] if digests else getattr(image, "id", None)
        except Exception as exc:
            raise ReplayError(f"Could not inspect prepared image {self.resolved.image}: {exc}") from exc
        name = f"replaylab-{_safe_name(self.resolved.instance_id)}-{uuid4().hex[:8]}"
        try:
            self.container = self.client.containers.create(
                image=self.resolved.image,
                name=name,
                command=["tail", "-f", "/dev/null"],
                detach=True,
                working_dir="/testbed",
                network_disabled=not self.allow_network,
                stdin_open=False,
                tty=False,
                privileged=False,
                volumes={},
            )
            self.container.start()
        except Exception as exc:
            self.cleanup()
            raise ReplayError(f"Could not start replay container: {exc}") from exc

    def run_action(self, action: TrajectoryAction) -> ExecResult:
        if self.container is None:
            raise ReplayError("Replay container has not been started")
        argv = [
            "timeout",
            "--signal=TERM",
            "--kill-after=2",
            str(self.timeout),
            *self.settings.interpreter,
            action.command,
        ]
        started = time.monotonic()
        try:
            raw = self.container.exec_run(
                argv,
                workdir=self.settings.cwd,
                environment=self.settings.env,
                stdout=True,
                stderr=True,
                demux=False,
            )
        except Exception as exc:
            raise ReplayError(f"Docker exec failed at step {action.step}: {exc}") from exc
        duration = time.monotonic() - started
        returncode, output = _unpack_exec_result(raw)
        return ExecResult(
            stdout=_decode_output(output),
            returncode=returncode,
            timed_out=returncode in (124, 137),
            duration_seconds=duration,
        )

    def exec_internal(
        self,
        argv: Sequence[str],
        *,
        cwd: str = "/testbed",
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self.container is None:
            raise ReplayError("Replay container has not been started")
        try:
            raw = self.container.exec_run(
                list(argv),
                workdir=cwd,
                environment=env or {},
                stdout=True,
                stderr=True,
                demux=False,
            )
        except Exception as exc:
            raise ReplayError(f"Internal container command failed: {exc}") from exc
        returncode, output = _unpack_exec_result(raw)
        return ExecResult(stdout=_decode_output(output), returncode=returncode)

    def cleanup(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            finally:
                self.container = None

    def _ensure_image(self) -> None:
        assert self.client is not None
        try:
            self.client.images.get(self.resolved.image)
            self.image_source = "local"
            return
        except Exception as exc:
            if not _is_not_found(exc):
                raise ReplayError(f"Could not inspect Docker image {self.resolved.image}: {exc}") from exc

        self.progress(f"Pulling official SWE-bench image {self.resolved.image}")
        try:
            self.client.images.pull(self.resolved.image)
            self.image_source = "pulled"
            return
        except Exception as pull_error:
            if not _is_not_found(pull_error):
                raise ReplayError(f"Could not pull image {self.resolved.image}: {pull_error}") from pull_error

        self.progress("Published image missing; building from the official SWE-bench task repository")
        try:
            _build_official_image(self.resolved, self.client)
            self.client.images.get(self.resolved.image)
            self.image_source = "built"
        except Exception as build_error:
            raise ReplayError(
                f"Image {self.resolved.image} is unavailable and could not be built from the "
                "official SWE-bench task repository. "
                f"Pull error: {pull_error}; build error: {build_error}"
            ) from build_error


def _build_official_image(resolved: ResolvedInstance, client: Any) -> None:
    if resolved.dataset not in {
        "SWE-bench/SWE-bench",
        "SWE-bench/SWE-bench_Lite",
        "SWE-bench/SWE-bench_Verified",
    }:
        raise ReplayError(
            "Automatic image building is only supported for official SWE-bench public datasets"
        )

    from swebench.image_builder.docker_build import build_instance_image
    from swebench.image_builder.image_spec import get_image_specs_from_dataset
    from swebench.task.repo import load_dockerfiles, task_paths

    image_name = resolved.image.removeprefix("docker.io/")
    image_repository, _, tag = image_name.rpartition(":")
    if not image_repository or not tag or "/" not in image_repository:
        raise ReplayError(f"Cannot derive build namespace/tag from image {resolved.image}")
    namespace = image_repository.rsplit("/", 1)[0]
    if resolved.image.startswith("docker.io/"):
        namespace = "docker.io/" + namespace

    with tempfile.TemporaryDirectory(prefix="replaylab-swebench-tasks-") as directory:
        repo_path = Path(directory)
        clone = subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/SWE-bench/swe-bench-tasks.git",
                str(repo_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if clone.returncode != 0:
            raise ReplayError(f"Could not clone official task repository: {clone.stdout.strip()}")
        ids = [resolved.instance_id]
        dockerfiles = load_dockerfiles(repo_path, ids)
        contexts = task_paths(repo_path, ids)
        specs = get_image_specs_from_dataset(
            [resolved.dataset_row], dockerfiles, namespace, tag, contexts
        )
        if len(specs) != 1 or specs[0].name != resolved.image.lower():
            raise ReplayError(
                "Official builder generated an image name different from the dataset image: "
                f"{specs[0].name if specs else '<none>'} != {resolved.image}"
            )
        build_instance_image(specs[0], client, logger=None, nocache=False)


def replay_trajectory(
    parsed: ParsedTrajectory,
    resolved: ResolvedInstance,
    trajectory_path: str | Path,
    *,
    allow_network: bool = False,
    timeout_override: int | None = None,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    started_at = _now()
    result = AnalysisResult(
        instance_id=resolved.instance_id,
        dataset=resolved.dataset,
        split=resolved.split,
        image=resolved.image,
        base_commit=resolved.base_commit,
        trajectory_path=str(Path(trajectory_path)),
        started_at=started_at,
        recorded_image=parsed.recorded_image,
    )
    runner = DockerRunner(
        resolved,
        parsed.settings,
        allow_network=allow_network,
        timeout_override=timeout_override,
        client=client,
        progress=progress,
    )
    announce = progress or (lambda _message: None)
    try:
        if parsed.recorded_image and _canonical_image(parsed.recorded_image) != _canonical_image(resolved.image):
            raise ReplayError(
                "Trajectory image does not match the official dataset instance image: "
                f"{parsed.recorded_image} != {resolved.image}"
            )
        runner.start()
        result.image_digest = runner.image_digest
        result.image_source = runner.image_source
        if runner.image_source == "built":
            result.warnings.append(
                "Published image was unavailable; this image was rebuilt from the current "
                "official task repository. Its digest may differ from the trajectory's original environment."
            )
        tracker = StateTracker(runner)
        baseline = tracker.snapshot()
        result.baseline = baseline
        result.warnings.extend(tracker.verify_base_commit(resolved.base_commit, baseline))
        current = baseline

        for action in parsed.actions:
            announce(f"Step {action.step}/{len(parsed.actions)}: {first_line(action.command)}")
            step_started = time.monotonic()
            try:
                outcome = runner.run_action(action)
            except ReplayError as exc:
                result.steps.append(
                    StepResult(
                        action=action,
                        replay_stdout="",
                        replay_returncode=None,
                        timed_out=False,
                        duration_seconds=time.monotonic() - step_started,
                        output_match=None,
                        before=current,
                        after=None,
                        error=str(exc),
                    )
                )
                result.error = str(exc)
                break

            try:
                after = tracker.snapshot()
                changes, worktree_diff, index_diff = tracker.compare(current, after)
            except (StateError, ReplayError) as exc:
                result.steps.append(
                    StepResult(
                        action=action,
                        replay_stdout=outcome.stdout,
                        replay_returncode=outcome.returncode,
                        timed_out=outcome.timed_out,
                        duration_seconds=outcome.duration_seconds,
                        output_match=compare_recorded_output(action, outcome.stdout),
                        before=current,
                        after=None,
                        error=str(exc),
                    )
                )
                result.error = f"State tracking failed after step {action.step}: {exc}"
                break

            step = StepResult(
                action=action,
                replay_stdout=outcome.stdout,
                replay_returncode=outcome.returncode,
                timed_out=outcome.timed_out,
                duration_seconds=outcome.duration_seconds,
                output_match=compare_recorded_output(action, outcome.stdout),
                before=current,
                after=after,
                changes=changes,
                worktree_diff=worktree_diff,
                index_diff=index_diff,
                signals=detect_signals(changes),
            )
            result.steps.append(step)
            current = after

        result.complete = result.error is None and len(result.steps) == len(parsed.actions)
    except (ReplayError, StateError) as exc:
        result.error = str(exc)
    finally:
        runner.cleanup()
        result.completed_at = _now()
        result.summary = summarize(result, len(parsed.actions))
    return result


_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)


def normalize_output(value: str) -> str:
    return _ANSI_ESCAPE.sub("", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def compare_recorded_output(action: TrajectoryAction, replay_stdout: str) -> bool | None:
    if action.recorded_stdout is None:
        return None
    return normalize_output(action.recorded_stdout) == normalize_output(replay_stdout)


def summarize(result: AnalysisResult, expected_steps: int) -> dict[str, Any]:
    comparable = [step for step in result.steps if step.output_match is not None]
    matches = sum(step.output_match is True for step in comparable)
    signal_counts = {
        signal: sum(signal in step.signals for step in result.steps)
        for signal in (
            "TEST_FILE_MODIFIED",
            "TEST_FILE_DELETED",
            "TEST_CONFIG_MODIFIED",
        )
    }
    return {
        "steps_total": expected_steps,
        "steps_replayed": sum(
            step.replay_returncode is not None for step in result.steps
        ),
        "recorded_outputs_available": len(comparable),
        "normalized_output_matches": matches,
        "output_match_rate": matches / len(comparable) if comparable else None,
        "commands_timed_out": sum(step.timed_out for step in result.steps),
        "commands_nonzero": sum(
            step.replay_returncode not in (None, 0) for step in result.steps
        ),
        "steps_with_repository_changes": sum(bool(step.changes) for step in result.steps),
        "signal_counts": signal_counts,
        "complete": result.complete,
    }


def first_line(command: str, limit: int = 100) -> str:
    value = command.strip().splitlines()[0] if command.strip() else ""
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _unpack_exec_result(raw: Any) -> tuple[int, Any]:
    if hasattr(raw, "exit_code"):
        return int(raw.exit_code), raw.output
    if isinstance(raw, tuple) and len(raw) == 2:
        return int(raw[0]), raw[1]
    raise ReplayError(f"Unexpected Docker exec result: {type(raw).__name__}")


def _decode_output(output: Any) -> str:
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if output is None:
        return ""
    if isinstance(output, tuple):
        return "".join(_decode_output(part) for part in output if part is not None)
    return str(output)


def _is_not_found(exc: Exception) -> bool:
    return exc.__class__.__name__ == "ImageNotFound" or getattr(exc, "status_code", None) == 404


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.").lower()[:80] or "task"


def _canonical_image(image: str) -> str:
    return image.removeprefix("docker.io/").lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
