from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from replaylab.analysis import AnalysisArtifacts
from replaylab.state import AnalysisResult
from replaylab.trajectory import TrajectoryError
from replaylab.ui import create_app, run_ui


def trajectory_bytes() -> bytes:
    return json.dumps({
        "trajectory_format": "mini-swe-agent-1.1",
        "info": {"config": {"environment": {}}},
        "messages": [{"role": "assistant", "extra": {"actions": [{"command": "pwd"}]}}],
    }).encode("utf-8")


def submit(client, **overrides):
    data = {
        "trajectory": (BytesIO(trajectory_bytes()), "sample.traj.json"),
        "instance": "example__repo-1",
        "dataset": "dataset",
        "split": "test",
        "timeout": "17",
    }
    data.update(overrides)
    return client.post("/api/run", data=data, content_type="multipart/form-data")


def wait_for_state(client, expected, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        payload = client.get("/api/status").get_json()
        if payload["status"] == expected:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"UI did not reach {expected}")


def successful_analyzer(complete=True, error=None):
    def analyze(_trajectory, instance, html_path, **kwargs):
        kwargs["progress"]("Step 1/1: pwd")
        result = AnalysisResult(
            instance,
            kwargs["dataset"],
            kwargs["split"],
            "image",
            "base",
            kwargs["trajectory_label"],
            "start",
            completed_at="end",
            complete=complete,
            error=error,
            summary={
                "steps_total": 1,
                "steps_replayed": 1,
                "normalized_output_matches": 1,
                "recorded_outputs_available": 1,
                "steps_with_repository_changes": 1,
                "commands_timed_out": 0,
                "signal_counts": {"TEST_FILE_MODIFIED": 1},
            },
        )
        html = Path(html_path)
        json_path = Path(kwargs["json_path"])
        html.write_text("<html>report</html>", encoding="utf-8")
        json_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        return AnalysisArtifacts(result, html, json_path)
    return analyze


class UITests(unittest.TestCase):
    def test_launcher_and_idle_status(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, successful_analyzer())
            client = app.test_client()
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"Analyze one trajectory", page.data)
            status = client.get("/api/status").get_json()
            self.assertEqual(status["status"], "idle")
            self.assertIsNone(status["report_url"])

    def test_validates_required_fields_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, successful_analyzer()).test_client()
            missing_file = client.post("/api/run", data={"instance": "x"})
            self.assertEqual(missing_file.status_code, 400)
            missing_instance = submit(client, instance="")
            self.assertEqual(missing_instance.status_code, 400)
            bad_timeout = submit(client, timeout="zero")
            self.assertEqual(bad_timeout.status_code, 400)
            self.assertIn("positive whole number", bad_timeout.get_json()["error"])

    def test_successful_run_exposes_only_current_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, successful_analyzer())
            client = app.test_client()
            response = submit(client, allow_network="on")
            self.assertEqual(response.status_code, 202)
            status = wait_for_state(client, "complete")
            self.assertTrue(status["result_complete"])
            self.assertEqual(status["phase"], "complete")
            self.assertEqual(status["summary"]["steps_replayed"], 1)
            rendered = client.get("/report")
            self.assertEqual(rendered.status_code, 200)
            rendered.close()
            downloaded = client.get("/report.json")
            self.assertEqual(downloaded.status_code, 200)
            self.assertIn("attachment", downloaded.headers["Content-Disposition"])
            downloaded.close()
            self.assertEqual(client.get("/report/anything").status_code, 404)

    def test_partial_replay_keeps_report_available(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(directory, successful_analyzer(False, "container stopped"))
            client = app.test_client()
            self.assertEqual(submit(client).status_code, 202)
            status = wait_for_state(client, "complete")
            self.assertFalse(status["result_complete"])
            self.assertEqual(status["error"], "container stopped")
            rendered = client.get("/report")
            self.assertEqual(rendered.status_code, 200)
            rendered.close()

    def test_pre_replay_failure_is_reported(self):
        def failing_analyzer(*_args, **_kwargs):
            raise TrajectoryError("Unsupported trajectory format")

        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, failing_analyzer).test_client()
            self.assertEqual(submit(client).status_code, 202)
            status = wait_for_state(client, "failed")
            self.assertIn("Unsupported trajectory format", status["error"])
            self.assertEqual(client.get("/report").status_code, 404)

    def test_rejects_concurrent_run_and_preserves_unrelated_output(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_analyzer(*args, **kwargs):
            kwargs["progress"]("Step 1/1: pwd")
            entered.set()
            release.wait(3)
            return successful_analyzer()(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.html").write_text("old", encoding="utf-8")
            (output / "report.json").write_text("old", encoding="utf-8")
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            client = create_app(output, blocking_analyzer).test_client()
            self.assertEqual(submit(client).status_code, 202)
            self.assertTrue(entered.wait(1))
            running = client.get("/api/status").get_json()
            self.assertEqual(running["phase"], "replaying")
            self.assertEqual(running["message"], "Step 1/1: pwd")
            self.assertFalse((output / "report.html").exists())
            self.assertFalse((output / "report.json").exists())
            self.assertTrue((output / "keep.txt").exists())
            second = submit(client)
            self.assertEqual(second.status_code, 409)
            release.set()
            wait_for_state(client, "complete")

    def test_stale_files_are_not_served_and_upload_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.html").write_text("stale", encoding="utf-8")
            app = create_app(output, successful_analyzer())
            client = app.test_client()
            self.assertEqual(client.get("/report").status_code, 404)
            app.config["MAX_CONTENT_LENGTH"] = 128
            oversized = submit(client)
            self.assertEqual(oversized.status_code, 413)
            self.assertIn("64 MB", oversized.get_json()["error"])

    def test_run_ui_binds_localhost_and_controls_browser_opening(self):
        app = Mock()
        timer = Mock()
        with (
            patch("replaylab.ui.create_app", return_value=app),
            patch("replaylab.ui.threading.Timer", return_value=timer) as timer_factory,
            patch("builtins.print"),
        ):
            run_ui(port=9001, output_dir="output", open_browser=True)
        timer_factory.assert_called_once()
        timer.start.assert_called_once_with()
        app.run.assert_called_once_with(
            host="127.0.0.1",
            port=9001,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

        no_open_app = Mock()
        with (
            patch("replaylab.ui.create_app", return_value=no_open_app),
            patch("replaylab.ui.threading.Timer") as no_open_timer,
            patch("builtins.print"),
        ):
            run_ui(port=9002, output_dir="output", open_browser=False)
        no_open_timer.assert_not_called()
