from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from replaylab.docker_runner import replay_trajectory, resolve_instance
from replaylab.report import write_reports
from replaylab.state import AnalysisResult
from replaylab.trajectory import load_trajectory


DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"


@dataclass(frozen=True)
class AnalysisArtifacts:
    result: AnalysisResult
    html_path: Path
    json_path: Path


def run_analysis(
    trajectory_path: str | Path,
    instance_id: str,
    html_path: str | Path,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = "test",
    timeout: int | None = None,
    allow_network: bool = False,
    json_path: str | Path | None = None,
    trajectory_label: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnalysisArtifacts:
    announce = progress or (lambda _message: None)
    announce("Loading trajectory")
    parsed = load_trajectory(trajectory_path)
    announce(f"Loaded {len(parsed.actions)} actions")
    announce(f"Resolving {instance_id} in {dataset}/{split}")
    resolved = resolve_instance(dataset, split, instance_id)
    announce("Preparing replay environment")
    result = replay_trajectory(
        parsed,
        resolved,
        trajectory_path,
        allow_network=allow_network,
        timeout_override=timeout,
        progress=announce,
    )
    if trajectory_label is not None:
        result.trajectory_path = trajectory_label
    announce("Writing report")
    written_html, written_json = write_reports(result, html_path, json_path)
    announce("Report ready")
    return AnalysisArtifacts(result, written_html, written_json)
