from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from replaylab.docker_runner import (
    DockerRunner,
    ExecResult,
    ReplayError,
    ResolvedInstance,
    _build_official_image,
    compare_recorded_output,
    normalize_output,
    replay_trajectory,
    resolve_instance,
)
from replaylab.state import RepoSnapshot
from replaylab.trajectory import ExecutionSettings, ParsedTrajectory, TrajectoryAction


class ImageNotFound(Exception):
    pass


class FakeImages:
    def __init__(self, initially_present=True, pull_fails=False):
        self.present = initially_present
        self.pull_fails = pull_fails
        self.pulled = []

    def get(self, name):
        if not self.present:
            raise ImageNotFound(name)
        return object()

    def pull(self, name):
        self.pulled.append(name)
        if self.pull_fails:
            raise ImageNotFound(name)
        self.present = True
        return object()


class FakeExecResult:
    def __init__(self, exit_code, output):
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    def __init__(self):
        self.started = False
        self.removed = False
        self.exec_calls = []
        self.next_exit_code = 0
        self.next_output = b"ok\n"

    def start(self):
        self.started = True

    def exec_run(self, argv, **kwargs):
        self.exec_calls.append((argv, kwargs))
        return FakeExecResult(self.next_exit_code, self.next_output)

    def remove(self, force=False):
        self.removed = force


class FailingStartContainer(FakeContainer):
    def start(self):
        raise RuntimeError("start failed")


class FakeContainers:
    def __init__(self):
        self.kwargs = None
        self.container = FakeContainer()

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.container


class FakeClient:
    def __init__(self, image_present=True, pull_fails=False):
        self.images = FakeImages(image_present, pull_fails)
        self.containers = FakeContainers()


class FakeTracker:
    snapshot_value = RepoSnapshot("abc", "tree", "tree", "")

    def __init__(self, _runner):
        pass

    def snapshot(self):
        return self.snapshot_value

    def compare(self, _before, _after):
        return [], "", ""

    def verify_base_commit(self, _base_commit, _snapshot):
        return []


def resolved():
    return ResolvedInstance(
        instance_id="django__django-12345",
        dataset="SWE-bench/SWE-bench_Verified",
        split="test",
        image="swebench/example:latest",
        base_commit="abc",
        dataset_row={},
        test_spec=object(),
    )


class DockerRunnerTests(unittest.TestCase):
    def test_resolves_row_and_image_through_swebench_5_api(self):
        row = {
            "instance_id": "django__django-12345",
            "base_commit": "deadbeef",
            "image": "docker.io/swebench/example:latest",
        }
        utilities = types.ModuleType("swebench.harness.utils")
        utilities.load_swebench_dataset = Mock(return_value=[row])
        utilities.make_test_spec = Mock(return_value=SimpleNamespace(image=row["image"]))
        modules = {
            "swebench": types.ModuleType("swebench"),
            "swebench.harness": types.ModuleType("swebench.harness"),
            "swebench.harness.utils": utilities,
        }
        with patch.dict(sys.modules, modules):
            result = resolve_instance(
                "SWE-bench/SWE-bench_Verified", "test", "django__django-12345"
            )
        self.assertEqual(result.image, row["image"])
        self.assertEqual(result.base_commit, "deadbeef")
        utilities.load_swebench_dataset.assert_called_once_with(
            "SWE-bench/SWE-bench_Verified", "test", ["django__django-12345"]
        )

    def test_normalization(self):
        self.assertEqual(normalize_output("\x1b[31mhello\x1b[0m\r\n"), "hello")
        self.assertEqual(normalize_output("\x1b]0;title\x07hello\r\n"), "hello")
        action = TrajectoryAction(1, "echo hello", "hello\n", 0)
        self.assertTrue(compare_recorded_output(action, " hello\r\n"))
        self.assertIsNone(compare_recorded_output(TrajectoryAction(1, "true"), ""))

    def test_secure_container_and_recorded_execution_settings(self):
        client = FakeClient()
        settings = ExecutionSettings(
            cwd="/testbed",
            interpreter=("bash", "-c"),
            env={"PAGER": "cat"},
            timeout=41,
        )
        runner = DockerRunner(resolved(), settings, client=client)
        runner.start()
        create = client.containers.kwargs
        self.assertTrue(create["network_disabled"])
        self.assertFalse(create["privileged"])
        self.assertEqual(create["volumes"], {})

        outcome = runner.run_action(TrajectoryAction(1, "echo ok"))
        argv, kwargs = client.containers.container.exec_calls[-1]
        self.assertEqual(
            argv[:5], ["timeout", "--signal=TERM", "--kill-after=2", "41", "bash"]
        )
        self.assertEqual(argv[-1], "echo ok")
        self.assertEqual(kwargs["environment"], {"PAGER": "cat"})
        self.assertEqual(outcome.stdout, "ok\n")
        runner.cleanup()
        self.assertTrue(client.containers.container.removed)

    def test_pulls_missing_image(self):
        client = FakeClient(image_present=False)
        runner = DockerRunner(resolved(), ExecutionSettings(), client=client)
        runner.start()
        self.assertEqual(client.images.pulled, ["swebench/example:latest"])
        self.assertEqual(runner.image_source, "pulled")
        runner.cleanup()

    def test_builds_when_published_image_is_missing(self):
        client = FakeClient(image_present=False, pull_fails=True)

        def built(_resolved, _client):
            client.images.present = True

        with patch("replaylab.docker_runner._build_official_image", side_effect=built) as build:
            runner = DockerRunner(resolved(), ExecutionSettings(), client=client)
            runner.start()
            build.assert_called_once()
            self.assertEqual(runner.image_source, "built")
            runner.cleanup()
        self.assertTrue(client.containers.container.removed)

    def test_official_builder_uses_swebench_5_image_spec_api(self):
        image_spec = SimpleNamespace(name="swebench/example:latest")
        image_specs = Mock(return_value=[image_spec])
        build = Mock()
        docker_build_module = types.ModuleType("swebench.image_builder.docker_build")
        docker_build_module.build_instance_image = build
        image_spec_module = types.ModuleType("swebench.image_builder.image_spec")
        image_spec_module.get_image_specs_from_dataset = image_specs
        repo_module = types.ModuleType("swebench.task.repo")
        repo_module.load_dockerfiles = Mock(return_value={"django__django-12345": "FROM scratch"})
        repo_module.task_paths = Mock(return_value={"django__django-12345": "/task/context"})
        modules = {
            "swebench": types.ModuleType("swebench"),
            "swebench.image_builder": types.ModuleType("swebench.image_builder"),
            "swebench.image_builder.docker_build": docker_build_module,
            "swebench.image_builder.image_spec": image_spec_module,
            "swebench.task": types.ModuleType("swebench.task"),
            "swebench.task.repo": repo_module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch("replaylab.docker_runner.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout="cloned"
            )) as clone,
        ):
            _build_official_image(resolved(), FakeClient())
        self.assertEqual(clone.call_args.args[0][:3], ["git", "clone", "--depth"])
        self.assertEqual(image_specs.call_args.args[2:4], ("swebench", "latest"))
        build.assert_called_once()

    def test_timeout_exit_is_marked_and_does_not_raise(self):
        client = FakeClient()
        runner = DockerRunner(resolved(), ExecutionSettings(), client=client)
        runner.start()
        client.containers.container.next_exit_code = 124
        client.containers.container.next_output = b"partial"
        outcome = runner.run_action(TrajectoryAction(1, "pytest"))
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.returncode, 124)
        runner.cleanup()

    def test_force_killed_timeout_exit_is_marked(self):
        client = FakeClient()
        runner = DockerRunner(resolved(), ExecutionSettings(), client=client)
        runner.start()
        client.containers.container.next_exit_code = 137
        outcome = runner.run_action(TrajectoryAction(1, "stubborn command"))
        self.assertTrue(outcome.timed_out)
        runner.cleanup()

    def test_allow_network_is_explicit(self):
        client = FakeClient()
        runner = DockerRunner(
            resolved(), ExecutionSettings(), allow_network=True, client=client
        )
        runner.start()
        self.assertFalse(client.containers.kwargs["network_disabled"])
        runner.cleanup()

    def test_start_failure_removes_created_container(self):
        client = FakeClient()
        client.containers.container = FailingStartContainer()
        runner = DockerRunner(resolved(), ExecutionSettings(), client=client)
        with self.assertRaisesRegex(Exception, "Could not start replay container"):
            runner.start()
        self.assertTrue(client.containers.container.removed)
        self.assertIsNone(runner.container)

    def test_replay_continues_after_timeout_and_cleans_up(self):
        client = FakeClient()
        parsed = ParsedTrajectory(
            "mini-swe-agent-1.1",
            (
                TrajectoryAction(1, "slow command", "partial", 124),
                TrajectoryAction(2, "true", "", 0),
            ),
            ExecutionSettings(),
        )
        outcomes = [
            ExecResult("partial", 124, True, 1.0),
            ExecResult("", 0, False, 0.1),
        ]
        with (
            patch("replaylab.docker_runner.StateTracker", FakeTracker),
            patch.object(DockerRunner, "run_action", side_effect=outcomes),
        ):
            result = replay_trajectory(parsed, resolved(), "trajectory.json", client=client)
        self.assertTrue(result.complete)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(result.steps[0].timed_out)
        self.assertEqual(result.summary["commands_timed_out"], 1)
        self.assertTrue(client.containers.container.removed)

    def test_infrastructure_abort_produces_partial_result_and_cleans_up(self):
        client = FakeClient()
        parsed = ParsedTrajectory(
            "mini-swe-agent-1.1",
            (TrajectoryAction(1, "echo never"),),
            ExecutionSettings(),
        )
        with (
            patch("replaylab.docker_runner.StateTracker", FakeTracker),
            patch.object(
                DockerRunner, "run_action", side_effect=ReplayError("daemon disconnected")
            ),
        ):
            result = replay_trajectory(parsed, resolved(), "trajectory.json", client=client)
        self.assertFalse(result.complete)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.summary["steps_replayed"], 0)
        self.assertIn("daemon disconnected", result.error or "")
        self.assertTrue(client.containers.container.removed)

    def test_mismatched_recorded_image_fails_before_container_creation(self):
        client = FakeClient()
        parsed = ParsedTrajectory(
            "mini-swe-agent-1.1",
            (TrajectoryAction(1, "pwd"),),
            ExecutionSettings(),
            recorded_image="docker.io/other/repo:latest",
        )
        result = replay_trajectory(parsed, resolved(), "trajectory.json", client=client)
        self.assertFalse(result.complete)
        self.assertIn("does not match", result.error or "")
        self.assertIsNone(client.containers.kwargs)

    def test_docker_io_prefix_does_not_create_false_image_mismatch(self):
        client = FakeClient()
        parsed = ParsedTrajectory(
            "mini-swe-agent-1.1",
            (TrajectoryAction(1, "pwd", "ok\n", 0),),
            ExecutionSettings(),
            recorded_image="docker.io/swebench/example:latest",
        )
        with patch("replaylab.docker_runner.StateTracker", FakeTracker):
            result = replay_trajectory(parsed, resolved(), "trajectory.json", client=client)
        self.assertTrue(result.complete, result.error)
        self.assertEqual(result.recorded_image, parsed.recorded_image)


if __name__ == "__main__":
    unittest.main()
