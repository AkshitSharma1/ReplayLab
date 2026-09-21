from __future__ import annotations

import shutil
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from replaylab.analysis import AnalysisArtifacts, DEFAULT_DATASET, run_analysis


MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RunRequest:
    trajectory_path: Path
    trajectory_name: str
    temporary_dir: Path
    instance_id: str
    dataset: str
    split: str
    timeout: int | None
    allow_network: bool


class UIController:
    def __init__(
        self,
        output_dir: str | Path,
        analyzer: Callable[..., AnalysisArtifacts] = run_analysis,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.html_path = self.output_dir / "report.html"
        self.json_path = self.output_dir / "report.json"
        self.analyzer = analyzer
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.status = "idle"
        self.phase = "idle"
        self.message = "Ready to replay a trajectory"
        self.summary: dict[str, object] | None = None
        self.result_complete: bool | None = None
        self.error: str | None = None

    def submit(self, run: RunRequest) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.html_path.unlink(missing_ok=True)
            self.json_path.unlink(missing_ok=True)
            self.status = "running"
            self.phase = "preparing"
            self.message = "Preparing uploaded trajectory"
            self.summary = None
            self.result_complete = None
            self.error = None
            self.thread = threading.Thread(
                target=self._execute,
                args=(run,),
                name="replaylab-ui-run",
                daemon=False,
            )
            self.thread.start()
            return True

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            available = self.status == "complete"
            return {
                "status": self.status,
                "phase": self.phase,
                "message": self.message,
                "summary": self.summary,
                "result_complete": self.result_complete,
                "error": self.error,
                "report_url": "/report" if available else None,
                "json_url": "/report.json" if available else None,
            }

    def report_path(self, kind: str) -> Path | None:
        with self.lock:
            if self.status != "complete":
                return None
            path = self.html_path if kind == "html" else self.json_path
            return path if path.is_file() else None

    def _progress(self, message: str) -> None:
        with self.lock:
            self.message = message
            self.phase = _phase_for(message)

    def _execute(self, run: RunRequest) -> None:
        try:
            artifacts = self.analyzer(
                run.trajectory_path,
                run.instance_id,
                self.html_path,
                dataset=run.dataset,
                split=run.split,
                timeout=run.timeout,
                allow_network=run.allow_network,
                json_path=self.json_path,
                trajectory_label=run.trajectory_name,
                progress=self._progress,
            )
            result = artifacts.result
            with self.lock:
                self.status = "complete"
                self.phase = "complete"
                self.message = "Replay complete" if result.complete else "Replay finished with errors"
                self.summary = result.summary
                self.result_complete = result.complete
                self.error = result.error
        except Exception as exc:
            with self.lock:
                self.status = "failed"
                self.phase = "failed"
                self.message = "Replay could not be completed"
                self.error = str(exc) or type(exc).__name__
        finally:
            shutil.rmtree(run.temporary_dir, ignore_errors=True)


def create_app(
    output_dir: str | Path,
    analyzer: Callable[..., AnalysisArtifacts] = run_analysis,
) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    controller = UIController(output_dir, analyzer)
    app.extensions["replaylab_controller"] = controller

    @app.get("/")
    def index():
        return render_template("ui.html.j2", default_dataset=DEFAULT_DATASET)

    @app.post("/api/run")
    def start_run():
        upload = request.files.get("trajectory")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a trajectory JSON file."), 400
        instance_id = request.form.get("instance", "").strip()
        if not instance_id:
            return jsonify(error="Enter a SWE-bench instance ID."), 400
        timeout_value = request.form.get("timeout", "").strip()
        try:
            timeout = _optional_positive_int(timeout_value)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        dataset = request.form.get("dataset", "").strip() or DEFAULT_DATASET
        split = request.form.get("split", "").strip() or "test"
        temporary_dir = Path(tempfile.mkdtemp(prefix="replaylab-ui-"))
        trajectory_path = temporary_dir / "trajectory.traj.json"
        try:
            upload.save(trajectory_path)
            if trajectory_path.stat().st_size == 0:
                raise ValueError("The trajectory file is empty.")
            run = RunRequest(
                trajectory_path=trajectory_path,
                trajectory_name=_display_name(upload.filename),
                temporary_dir=temporary_dir,
                instance_id=instance_id,
                dataset=dataset,
                split=split,
                timeout=timeout,
                allow_network=request.form.get("allow_network") in {"1", "true", "on"},
            )
            if not controller.submit(run):
                shutil.rmtree(temporary_dir, ignore_errors=True)
                return jsonify(error="A replay is already running."), 409
        except ValueError as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            return jsonify(error=str(exc)), 400
        except OSError as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            return jsonify(error=f"Could not store the uploaded trajectory: {exc}"), 500
        return jsonify(status="running"), 202

    @app.get("/api/status")
    def run_status():
        return jsonify(controller.snapshot())

    @app.get("/report")
    def html_report():
        path = controller.report_path("html")
        if path is None:
            abort(404)
        return send_file(path, mimetype="text/html")

    @app.get("/report.json")
    def json_report():
        path = controller.report_path("json")
        if path is None:
            abort(404)
        return send_file(
            path,
            mimetype="application/json",
            as_attachment=True,
            download_name="report.json",
        )

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify(error="Trajectory files must be 64 MB or smaller."), 413

    return app


def run_ui(
    *,
    port: int = 8765,
    output_dir: str | Path = "replaylab-output",
    open_browser: bool = True,
) -> None:
    app = create_app(output_dir)
    url = f"http://127.0.0.1:{port}/"
    print(f"ReplayLab UI: {url}")
    if open_browser:
        opener = threading.Timer(0.5, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False, threaded=True)


def _phase_for(message: str) -> str:
    lowered = message.lower()
    if lowered.startswith("resolving"):
        return "resolving"
    if lowered.startswith("step "):
        return "replaying"
    if "image" in lowered or "environment" in lowered:
        return "preparing_environment"
    if lowered.startswith("writing"):
        return "finalizing"
    return "preparing"


def _optional_positive_int(value: str) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("Timeout must be a positive whole number.") from exc
    if parsed <= 0:
        raise ValueError("Timeout must be a positive whole number.")
    return parsed


def _display_name(value: str) -> str:
    normalized = value.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] or "trajectory.traj.json"
