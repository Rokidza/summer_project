"""The Aim-backed `Tracker`. The only module in this repo that imports Aim.

Aim's repository is a collection of RocksDB databases. Those depend on POSIX
locking and mmap semantics that network and parallel filesystems handle badly,
so the repository belongs on local disk and nowhere else - never on Lustre, and
never on a compute node's shared scratch. A cluster-side finetune records to a
flat file instead (`jsonl_tracker.py`) and is replayed into Aim locally.

Browse what it wrote:

    aim up --repo .aim
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

DEFAULT_REPO = ".aim"
"""Local disk, beside the results database, and git-ignored."""


def _as_plotly(figure: Any):
    """Aim wants a plotly figure; a replayed run carries plotly *JSON*.

    Reconstructed here rather than in `replay`, so replay stays a pure push of
    recorded calls into a tracker and its tests need neither Aim nor plotly.
    """
    if not isinstance(figure, Mapping):
        return figure
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on the install
        # Aim's own Figure needs plotly too, so this is the whole figure path,
        # not just replay. Nothing in the platform logs one yet - the confusion
        # matrix is deliberately text - so it is worth saying which install is
        # missing rather than letting an import error stand on its own.
        raise ImportError(
            "replaying a figure needs plotly: uv pip install --python "
            ".venv/bin/python3 -e '.[train]'"
        ) from exc
    return go.Figure(dict(figure))


class AimTracker:
    """One Aim run, opened on construction and closed by `close`."""

    def __init__(
        self,
        *,
        repo: str | Path = DEFAULT_REPO,
        experiment: str = "router-finetune",
        name: str | None = None,
    ):
        from aim import Repo, Run

        # A bare directory is not an Aim repository - it has to be initialised,
        # or opening a run in it fails with an unexplained RuntimeError. Doing it
        # here means a first run needs no `aim init` step of its own.
        self._repo_path = str(Path(repo))
        if not Repo.exists(self._repo_path):
            Repo.from_path(self._repo_path, init=True)
        self._run = Run(repo=self._repo_path, experiment=experiment)
        if name:
            self._run.name = name

    @property
    def run_hash(self) -> str:
        """Aim's id for this run - what the UI's URL ends in."""
        return self._run.hash

    def log_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self._run[key] = value

    def log_metric(self, name, value, *, step=None, context=None) -> None:
        self._run.track(float(value), name=name, step=step, context=dict(context or {}))

    def log_figure(self, name, figure, *, step=None, context=None) -> None:
        from aim import Figure

        self._run.track(
            Figure(_as_plotly(figure)),
            name=name,
            step=step,
            context=dict(context or {}),
        )

    def log_text(self, name, text, *, step=None, context=None) -> None:
        from aim import Text

        self._run.track(
            Text(str(text)), name=name, step=step, context=dict(context or {})
        )

    def log_summary(self, key: str, value: Any) -> None:
        self._run[key] = value

    def close(self) -> None:
        if self._run is None:
            return
        run_hash = self._run.hash
        self._run.close()
        self._run = None
        self._index(run_hash)

    def _index(self, run_hash: str) -> None:
        """Fold a finished run into the repository's index.

        A run's data lands in its own chunk database; nothing can *read* it -
        not the UI, not a query - until it is indexed. `aim up` indexes on
        start, so skipping this only delays visibility, which is why a failure
        here warns rather than raises: the run's data is already on disk.
        """
        from aim import Repo
        from aim.sdk.index_manager import RepoIndexManager

        try:
            RepoIndexManager.get_index_manager(Repo.from_path(self._repo_path)).index(
                run_hash
            )
        except Exception as exc:  # pragma: no cover - depends on Aim internals
            print(f"warning: could not index Aim run {run_hash}: {exc}")
