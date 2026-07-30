"""A `Tracker` that keeps every call, so tests can assert what a run recorded.

The real backend is Aim, which the test suite never imports and never gives a
repository to: what training owes its caller is a sequence of tracker calls, and
that is exactly what this records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackedCall:
    """One `log_metric`/`log_figure`/`log_text` call, as it was made."""

    name: str
    value: Any
    step: int | None
    context: dict


@dataclass
class RecordingTracker:
    """Every call, in order, with nothing else attached."""

    params: dict = field(default_factory=dict)
    metrics: list[TrackedCall] = field(default_factory=list)
    figures: list[TrackedCall] = field(default_factory=list)
    texts: list[TrackedCall] = field(default_factory=list)
    summaries: dict = field(default_factory=dict)
    closed: bool = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)

    def log_metric(self, name, value, *, step=None, context=None) -> None:
        self.metrics.append(TrackedCall(name, value, step, dict(context or {})))

    def log_figure(self, name, figure, *, step=None, context=None) -> None:
        self.figures.append(TrackedCall(name, figure, step, dict(context or {})))

    def log_text(self, name, text, *, step=None, context=None) -> None:
        self.texts.append(TrackedCall(name, text, step, dict(context or {})))

    def log_summary(self, key: str, value: Any) -> None:
        self.summaries[key] = value

    def close(self) -> None:
        self.closed = True

    # -- assertions helpers ------------------------------------------------

    def metric_values(self, name: str, **context) -> list[float]:
        """Every value logged under `name` with exactly this context, in order."""
        return [
            call.value
            for call in self.metrics
            if call.name == name and call.context == context
        ]

    def contexts_of(self, name: str) -> list[dict]:
        return [call.context for call in self.metrics if call.name == name]
