from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing import QuijoteIndex


@dataclass(slots=True)
class IndexingWorkerResult:
    run_id: int
    path: Path
    stats: dict[str, int]
    index: QuijoteIndex
    nlp: Any


@dataclass(slots=True)
class ProgressSnapshot:
    run_id: int
    stage: str
    completed: int | None
    total: int | None
    updated_at: float
