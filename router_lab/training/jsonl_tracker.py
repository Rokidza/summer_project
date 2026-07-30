"""A `Tracker` that writes a flat file, and replay of that file into a tracker.

This exists for one reason: **Aim must never write its repository onto cluster
storage.** Its backend is a collection of RocksDB databases, which depend on
POSIX locking and mmap semantics that network and parallel filesystems handle
badly - the same class of hazard the platform already avoids by packaging
dependencies into a container image rather than unpacking a dependency tree onto
Lustre. So a cluster-side finetune records to one append-only file on the node's
own disk, the file comes home with the results database, and it is replayed into
the local Aim repository afterwards.

The payoff is that a cluster run and a laptop run sit side by side in one UI,
comparable on identical metrics, without a tracker reaching across the network
from a compute node.

Every call is flushed as it is made. A job killed at walltime costs the epoch it
was in, not the run: the closing record is a nicety, and replay does not need it.

Replay is refused rather than repeated. A receipt beside the file records what
was replayed where, so a second attempt says so instead of quietly producing a
duplicate run that would then be averaged into a comparison.

Nothing here imports Aim - it speaks the same protocol the training loop already
writes against, so which one a run uses is configuration.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from router_lab.training.tracker import Tracker

SCHEMA_VERSION = "1"
"""Bumped when a record's shape changes in a way replay cannot read back."""

RECEIPT_SUFFIX = ".replayed.json"

RUN = "run"
PARAMS = "params"
METRIC = "metric"
FIGURE = "figure"
TEXT = "text"
SUMMARY = "summary"
END = "end"


class ReplayRefused(Exception):
    """Raised rather than writing a run the destination repository already has."""


class JsonlTracker:
    """One training run, recorded as JSON lines.

    The header is written on construction, so a file exists - and says what it
    is - from before the first batch.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        experiment: str = "router-finetune",
        name: str | None = None,
        uid: str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.uid = uid or uuid.uuid4().hex
        self._handle = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "record": RUN,
                "schema": SCHEMA_VERSION,
                "uid": self.uid,
                "experiment": experiment,
                "name": name,
                "started_at": _now(),
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        """Append one record and flush it.

        `default=str` because a run must not die on its last line over a value
        that happened to be a Path: a coerced string in the tracker is a far
        better trade than a lost finetune. Figures are the one exception, and
        they are checked where they are logged.
        """
        if self._handle is None:
            raise ValueError(f"{self.path} is closed; the run is already recorded")
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._write({"record": PARAMS, "params": dict(params)})

    def log_metric(self, name, value, *, step=None, context=None) -> None:
        self._write(
            {
                "record": METRIC,
                "name": name,
                "value": float(value),
                "step": step,
                "context": dict(context or {}),
            }
        )

    def log_figure(self, name, figure, *, step=None, context=None) -> None:
        self._write(
            {
                "record": FIGURE,
                "name": name,
                "figure": _figure_payload(figure),
                "step": step,
                "context": dict(context or {}),
            }
        )

    def log_text(self, name, text, *, step=None, context=None) -> None:
        self._write(
            {
                "record": TEXT,
                "name": name,
                "text": str(text),
                "step": step,
                "context": dict(context or {}),
            }
        )

    def log_summary(self, key: str, value: Any) -> None:
        self._write({"record": SUMMARY, "key": key, "value": value})

    def close(self) -> None:
        if self._handle is None:
            return
        self._write({"record": END, "ended_at": _now()})
        self._handle.close()
        self._handle = None


def _figure_payload(figure: Any) -> dict:
    """A figure as plain JSON, or a refusal naming what was handed over.

    Coercing a figure to its repr would produce a file that replays into
    something unopenable, so this is the one place the writer is strict.
    """
    if hasattr(figure, "to_plotly_json"):
        return figure.to_plotly_json()
    if isinstance(figure, Mapping):
        return dict(figure)
    raise TypeError(
        f"cannot record a figure of type {type(figure).__name__}: it needs a "
        f"to_plotly_json() method, or to be a plain mapping of plotly JSON"
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass(frozen=True)
class RunHeader:
    """What a file says about the run it holds."""

    uid: str
    experiment: str
    name: str | None
    schema: str
    started_at: str


@dataclass(frozen=True)
class ReplayReport:
    """What one replay moved, for a caller that wants to print it."""

    uid: str
    header: RunHeader
    counts: dict[str, int]
    receipt_path: Path

    def summary_line(self) -> str:
        counted = ", ".join(
            f"{count} {kind}" for kind, count in sorted(self.counts.items())
        )
        return f"replayed {self.header.name or self.uid[:8]}: {counted}"


def read_records(path: str | Path) -> Iterator[dict]:
    """Every record in the file, in order, with the line number in any error."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} line {number} is not valid JSON: {exc}"
                ) from exc


def read_header(path: str | Path) -> RunHeader:
    """The run header, or a refusal to treat the file as a run at all."""
    first = next(read_records(path), None)
    if first is None:
        raise ValueError(f"{path} is empty: there is no run in it to replay")
    if first.get("record") != RUN:
        raise ValueError(
            f"{path} does not start with a run header (found "
            f"{first.get('record')!r}): it was not written by JsonlTracker"
        )
    return RunHeader(
        uid=first["uid"],
        experiment=first.get("experiment", "router-finetune"),
        name=first.get("name"),
        schema=first.get("schema", SCHEMA_VERSION),
        started_at=first.get("started_at", ""),
    )


def receipt_path_for(path: str | Path) -> Path:
    return Path(str(path) + RECEIPT_SUFFIX)


def ensure_replayable(path: str | Path, *, force: bool = False) -> None:
    """Raise `ReplayRefused` if this file has been replayed already.

    Separate from `replay` so a caller can check *before* opening a destination
    run: refusing halfway would leave an empty run behind in the repository.
    """
    receipt = receipt_path_for(path)
    if receipt.exists() and not force:
        raise ReplayRefused(
            f"{path} was already replayed (see {receipt}); replaying it again "
            f"would produce a duplicate run."
        )


def replay(
    path: str | Path,
    tracker: Tracker,
    *,
    force: bool = False,
    target: str | None = None,
) -> ReplayReport:
    """Push every recorded call into `tracker`, once.

    The tracker is not closed: its lifetime belongs to whoever opened it, the
    same convention the training loop follows.
    """
    path = Path(path)
    header = read_header(path)
    receipt = receipt_path_for(path)
    ensure_replayable(path, force=force)

    # Written before anything else: a replay interrupted half way leaves a run
    # in the destination, and it has to be recognisable as this file's - both to
    # a reader and to the duplicate check the CLI makes before it starts.
    replayed_at = _now()
    tracker.log_summary("replay/source", str(path))
    tracker.log_summary("replay/run_uid", header.uid)
    tracker.log_summary("replay/replayed_at", replayed_at)

    counts = {PARAMS: 0, METRIC: 0, FIGURE: 0, TEXT: 0, SUMMARY: 0}
    for record in read_records(path):
        kind = record.get("record")
        if kind in (RUN, END):
            continue
        if kind not in counts:
            raise ValueError(f"{path} holds an unknown record kind {kind!r}")
        counts[kind] += 1
        if kind == PARAMS:
            tracker.log_params(record["params"])
        elif kind == METRIC:
            tracker.log_metric(
                record["name"],
                record["value"],
                step=record.get("step"),
                context=record.get("context") or {},
            )
        elif kind == FIGURE:
            tracker.log_figure(
                record["name"],
                record["figure"],
                step=record.get("step"),
                context=record.get("context") or {},
            )
        elif kind == TEXT:
            tracker.log_text(
                record["name"],
                record["text"],
                step=record.get("step"),
                context=record.get("context") or {},
            )
        else:
            tracker.log_summary(record["key"], record["value"])

    receipt.write_text(
        json.dumps(
            {
                "uid": header.uid,
                "source": str(path),
                "target": target,
                "name": header.name,
                "experiment": header.experiment,
                "replayed_at": replayed_at,
                "counts": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReplayReport(
        uid=header.uid, header=header, counts=counts, receipt_path=receipt
    )
