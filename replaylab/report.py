from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from jinja2 import Environment, PackageLoader, select_autoescape

from replaylab.state import AnalysisResult


def write_reports(
    result: AnalysisResult,
    html_path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path]:
    destination = Path(html_path)
    sidecar = Path(json_path) if json_path is not None else destination.with_suffix(".json")
    if destination.resolve() == sidecar.resolve():
        raise ValueError("HTML and JSON output paths must be different")

    environment = Environment(
        loader=PackageLoader("replaylab", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html", "xml"), default=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html.j2")
    html = template.render(report=result)
    payload = json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
    _atomic_write(sidecar, payload)
    _atomic_write(destination, html)
    return destination, sidecar


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
