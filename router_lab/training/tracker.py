"""What a training run records, and where it records it.

One protocol, five kinds of call. The training loop is written against this and
nothing else, so where a run is recorded becomes configuration: Aim on a
laptop, and - once #19 lands - a flat file on a compute node, replayed into Aim
afterwards. Exactly one module in this package imports Aim.

A `context` is Aim's name for a label attached to a metric rather than to the
run, e.g. `{"subset": "train"}`. Metrics logged under the same name with
different contexts share a chart, which is the whole reason train and validation
curves are worth looking at.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """The recording surface of a training run."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record the run's configuration and provenance. Written once."""

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a number, optionally at a step and under a context."""

    def log_figure(
        self,
        name: str,
        figure: Any,
        *,
        step: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a plot - a confusion matrix, a distribution."""

    def log_text(
        self,
        name: str,
        text: str,
        *,
        step: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Record inspectable prose - the misroutes worth reading by eye."""

    def log_summary(self, key: str, value: Any) -> None:
        """Record a single final fact about the run: a checkpoint path, a score."""

    def close(self) -> None:
        """Finish the run. Safe to call twice."""


class NullTracker:
    """Records nothing. The default, so a training run needs no tracker to work."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metric(self, name, value, *, step=None, context=None) -> None:
        pass

    def log_figure(self, name, figure, *, step=None, context=None) -> None:
        pass

    def log_text(self, name, text, *, step=None, context=None) -> None:
        pass

    def log_summary(self, key: str, value: Any) -> None:
        pass

    def close(self) -> None:
        pass
