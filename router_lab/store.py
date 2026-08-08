"""The SQLite results store - the only module that knows the results schema.

Everything else in the platform (eval harness, label rule, dashboard) reads and
writes benchmark results through this module's functions. No other code opens
the database file or writes SQL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import astuple, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkResult:
    """One query answered by one approach, in one run.

    The natural key is (run_id, dataset, model, approach, query_id): re-running
    the same cell overwrites it rather than accumulating duplicates.
    """

    run_id: str
    dataset: str
    model: str
    approach: str
    query_id: str
    question: str
    correct: bool
    predicted: str
    gold: str
    latency_s: float
    completion_tokens: int
    total_tokens: int
    call_count: int
    error: str | None = None
    router_approach: str | None = None
    response: str = ""


@dataclass(frozen=True)
class SweepConfig:
    """How one (run, dataset, model) sweep was configured - its provenance.

    Recorded so a run's rows can be read years later without guessing how they
    were produced: how big the pool was, which approaches were asked for, and -
    for a two-stage sweep - the control fraction and seed that decided which
    already-solved queries stage two also covered.

    What is deliberately *not* here is per-query approach coverage: that is
    counted off the result rows themselves (`approach_coverage`), so it cannot
    drift away from them.
    """

    run_id: str
    dataset: str
    model: str
    mode: str
    """"single" (every approach over every query) or "two-stage"."""
    pool_size: int
    """Problems drawn for this dataset, i.e. how many stage one covered."""
    approaches: list[str]
    control_fraction: float | None = None
    """Two-stage only: the share of baseline-solved queries stage two also ran."""
    seed: int | None = None
    """Two-stage only: what made the control sample reproducible."""


SINGLE_STAGE = "single"
TWO_STAGE = "two-stage"

DEFAULT_DB = "results.sqlite"
DB_ENV_VAR = "ROUTER_LAB_DB"
"""Where to read/write results when no --db is given. Job scripts and the sync
step set the env var so neither the CLI nor the dashboard needs a path baked in."""

_COLUMNS = [f.name for f in fields(BenchmarkResult)]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    run_id            TEXT    NOT NULL,
    dataset           TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    approach          TEXT    NOT NULL,
    query_id          TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    correct           INTEGER NOT NULL,
    predicted         TEXT    NOT NULL,
    gold              TEXT    NOT NULL,
    latency_s         REAL    NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    call_count        INTEGER NOT NULL,
    error             TEXT,
    router_approach   TEXT,
    response          TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, dataset, model, approach, query_id)
);

CREATE TABLE IF NOT EXISTS sweep_configs (
    run_id           TEXT    NOT NULL,
    dataset          TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    mode             TEXT    NOT NULL,
    pool_size        INTEGER NOT NULL,
    approaches       TEXT    NOT NULL,
    control_fraction REAL,
    seed             INTEGER,
    PRIMARY KEY (run_id, dataset, model)
);
"""


class ResultsStore:
    """Read/write access to a results database at a given path.

    Safe to share across threads. Two callers need that: Streamlit runs every
    rerun on a fresh thread, and the harness writes from whichever thread is
    draining its worker pool. sqlite3 refuses cross-thread use by default, so
    the connection opts out of that check and a lock serialises every statement
    instead - which is all SQLite wants anyway, since a connection is a single
    transaction context.
    """

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @classmethod
    def open(cls, path: str | Path) -> "ResultsStore":
        """Open (creating if needed) the results database at `path`.

        Re-opening an existing database is safe: the schema is created only if
        absent, and existing rows are left untouched.
        """
        return cls(sqlite3.connect(str(path), check_same_thread=False))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _fetch(self, sql: str, params=()) -> list[sqlite3.Row]:
        """Run a query to completion under the lock; never hand out a cursor."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def write_result(self, result: BenchmarkResult) -> None:
        self.write_results([result])

    def write_results(self, results) -> None:
        rows = [astuple(r) for r in results]
        if not rows:
            return
        placeholders = ", ".join("?" * len(_COLUMNS))
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO results ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                rows,
            )
            self._conn.commit()

    def record_sweep_config(self, config: SweepConfig) -> None:
        """Record how a sweep was configured, replacing any earlier record of it.

        Written before the sweep runs, so a sweep that dies part-way still says
        what it set out to do - which is exactly when you want to know.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sweep_configs "
                "(run_id, dataset, model, mode, pool_size, approaches, "
                " control_fraction, seed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    config.run_id,
                    config.dataset,
                    config.model,
                    config.mode,
                    config.pool_size,
                    json.dumps(list(config.approaches)),
                    config.control_fraction,
                    config.seed,
                ),
            )
            self._conn.commit()

    # -- reading ----------------------------------------------------------

    def sweep_configs(
        self,
        *,
        run_id: str | None = None,
        dataset: str | None = None,
        model: str | None = None,
    ) -> list[SweepConfig]:
        """Recorded sweep configurations, one per (run, dataset, model) cell."""
        where, params = _where(run_id=run_id, dataset=dataset, model=model)
        rows = self._fetch(
            f"SELECT run_id, dataset, model, mode, pool_size, approaches, "
            f"control_fraction, seed FROM sweep_configs{where} "
            f"ORDER BY run_id, dataset, model",
            params,
        )
        return [
            SweepConfig(
                run_id=row["run_id"],
                dataset=row["dataset"],
                model=row["model"],
                mode=row["mode"],
                pool_size=row["pool_size"],
                approaches=json.loads(row["approaches"]),
                control_fraction=row["control_fraction"],
                seed=row["seed"],
            )
            for row in rows
        ]

    def approach_coverage(
        self, *, run_id: str, dataset: str, model: str
    ) -> dict[str, list[str]]:
        """Which approaches each query in one cell was actually run through.

        Derived by reading the rows, never stored: after a two-stage sweep the
        queries the baseline solved carry the baseline alone, while the rest
        carry the full set.
        """
        rows = self._fetch(
            "SELECT query_id, approach FROM results "
            "WHERE run_id = ? AND dataset = ? AND model = ? "
            "ORDER BY CAST(query_id AS INTEGER), query_id, approach",
            (run_id, dataset, model),
        )
        coverage: dict[str, list[str]] = {}
        for row in rows:
            coverage.setdefault(row["query_id"], []).append(row["approach"])
        return coverage


    def query_results(
        self,
        *,
        run_id: str | None = None,
        dataset: str | None = None,
        approach: str | None = None,
        model: str | None = None,
    ) -> list[BenchmarkResult]:
        """Results matching every filter supplied; unfiltered facets match all."""
        where, params = _where(
            run_id=run_id, dataset=dataset, approach=approach, model=model
        )
        rows = self._fetch(
            f"SELECT {', '.join(_COLUMNS)} FROM results{where} "
            f"ORDER BY run_id, dataset, model, query_id, approach",
            params,
        )
        return [_to_result(row) for row in rows]

    def results_for_query(
        self, *, run_id: str, dataset: str, model: str, query_id: str
    ) -> list[BenchmarkResult]:
        """Every approach's attempt at one query - the label rule's input."""
        where, params = _where(
            run_id=run_id, dataset=dataset, model=model, query_id=query_id
        )
        rows = self._fetch(
            f"SELECT {', '.join(_COLUMNS)} FROM results{where} ORDER BY approach",
            params,
        )
        return [_to_result(row) for row in rows]

    def aggregate_by_approach(
        self,
        *,
        run_id: str | None = None,
        dataset: str | None = None,
        approach: str | None = None,
        model: str | None = None,
    ) -> list["ApproachSummary"]:
        """Per-(approach, dataset, model) rollups, so callers write no SQL."""
        where, params = _where(
            run_id=run_id, dataset=dataset, approach=approach, model=model
        )
        rows = self._fetch(
            f"""
            SELECT approach, dataset, model,
                   COUNT(*)                        AS n,
                   SUM(correct)                    AS correct,
                   SUM(error IS NOT NULL)          AS errors,
                   AVG(latency_s)                  AS avg_latency_s,
                   AVG(completion_tokens)          AS avg_completion_tokens,
                   SUM(total_tokens)               AS total_tokens,
                   SUM(call_count)                 AS total_calls
            FROM results{where}
            GROUP BY approach, dataset, model
            ORDER BY dataset, model, approach
            """,
            params,
        )
        return [
            ApproachSummary(
                approach=row["approach"],
                dataset=row["dataset"],
                model=row["model"],
                n=row["n"],
                correct=row["correct"],
                accuracy=row["correct"] / row["n"] if row["n"] else 0.0,
                errors=row["errors"],
                avg_latency_s=row["avg_latency_s"],
                avg_completion_tokens=row["avg_completion_tokens"],
                total_tokens=row["total_tokens"],
                total_calls=row["total_calls"],
            )
            for row in rows
        ]

    def list_runs(self) -> list["RunInfo"]:
        """Every run in the store, oldest first, with what it covered."""
        rows = self._fetch(
            """
            SELECT run_id,
                   COUNT(*) AS n_results,
                   GROUP_CONCAT(DISTINCT dataset)  AS datasets,
                   GROUP_CONCAT(DISTINCT model)    AS models,
                   GROUP_CONCAT(DISTINCT approach) AS approaches
            FROM results
            GROUP BY run_id
            ORDER BY run_id
            """
        )
        return [
            RunInfo(
                run_id=row["run_id"],
                n_results=row["n_results"],
                datasets=sorted(row["datasets"].split(",")),
                models=sorted(row["models"].split(",")),
                approaches=sorted(row["approaches"].split(",")),
            )
            for row in rows
        ]

    def completed_query_ids(
        self, *, run_id: str, dataset: str, model: str, approach: str
    ) -> set[str]:
        """Query ids already durably recorded, without error, for one cell.

        Errored rows are excluded so a resumed sweep retries them instead of
        treating a prior failure as done - only a clean result counts as work
        that does not need repeating.
        """
        rows = self._fetch(
            "SELECT query_id FROM results "
            "WHERE run_id = ? AND dataset = ? AND model = ? AND approach = ? "
            "AND error IS NULL",
            (run_id, dataset, model, approach),
        )
        return {row["query_id"] for row in rows}

    def query_ids(
        self, *, run_id: str, dataset: str, model: str
    ) -> list[str]:
        """Distinct query ids in one (run, dataset, model) cell."""
        rows = self._fetch(
            "SELECT DISTINCT query_id FROM results "
            "WHERE run_id = ? AND dataset = ? AND model = ? "
            "ORDER BY CAST(query_id AS INTEGER), query_id",
            (run_id, dataset, model),
        )
        return [row["query_id"] for row in rows]


@dataclass(frozen=True)
class ApproachSummary:
    """Rollup of one (approach, dataset, model) cell."""

    approach: str
    dataset: str
    model: str
    n: int
    correct: int
    accuracy: float
    errors: int
    avg_latency_s: float
    avg_completion_tokens: float
    total_tokens: int
    total_calls: int


@dataclass(frozen=True)
class RunInfo:
    """What a single sweep covered."""

    run_id: str
    n_results: int
    datasets: list[str]
    models: list[str]
    approaches: list[str]


def _where(**filters) -> tuple[str, list]:
    """Build a conjunctive WHERE clause from the non-None filters."""
    active = {k: v for k, v in filters.items() if v is not None}
    if not active:
        return "", []
    clause = " AND ".join(f"{column} = ?" for column in active)
    return f" WHERE {clause}", list(active.values())


def _to_result(row: sqlite3.Row) -> BenchmarkResult:
    values = dict(row)
    values["correct"] = bool(values["correct"])
    return BenchmarkResult(**values)
