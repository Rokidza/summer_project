"""A passive Streamlit viewer over a results database synced down from Supek.

    streamlit run router_lab/dashboard.py -- --db results.sqlite

It never talks to the cluster and never writes SQL: everything it renders comes
from `router_lab.views`, which is where the logic (and the tests) live. This
file is presentation only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from router_lab.labels import FINETUNED_ROUTER, ROUTER
from router_lab.store import DB_ENV_VAR, DEFAULT_DB, ResultsStore
from router_lab.views import (
    QueryDetail,
    RouterComparison,
    comparisons_by_category,
    leaderboard,
    query_detail,
    router_comparison,
    routers_compared,
)

ROUTER_NAMES = {
    ROUTER: "optillm's pretrained router",
    FINETUNED_ROUTER: "this repo's finetuned router",
}
"""How each router is described on screen. Presentation, so it lives here."""


def router_name(router: str) -> str:
    return ROUTER_NAMES.get(router, router)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Where to read results from.

    `--db` wins; otherwise $ROUTER_LAB_DB, which is how job scripts and the
    sync step point the app at a database without editing a command line.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default=os.environ.get(DB_ENV_VAR, DEFAULT_DB),
        help=f"results database path (default: ${DB_ENV_VAR} or {DEFAULT_DB})",
    )
    args, _unknown = ap.parse_known_args(argv)
    return args


@st.cache_resource
def open_store(db_path: str, mtime: float) -> ResultsStore:
    """Keep one connection per (file, revision); a resynced file reopens itself."""
    del mtime  # only present so a changed file busts the cache
    return ResultsStore.open(db_path)


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def pick(label: str, options: list, *, key: str, **kwargs):
    """A selectbox that forgets a choice the current options no longer offer.

    Runs cover different datasets, models and query ids. Switching to a run
    without the previously selected one would otherwise leave a stale value in
    session state, which Streamlit rejects when it re-renders the widget.
    """
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return st.selectbox(label, options, key=key, **kwargs)


# -- panels ---------------------------------------------------------------


def render_leaderboard(store: ResultsStore, run_id: str, dataset, model) -> None:
    st.subheader("Leaderboard")
    rows = leaderboard(store, run_id=run_id, dataset=dataset, model=model)
    if not rows:
        st.info("No results for this selection.")
        return

    st.caption(
        "Cost is total tokens as a multiple of the `none` baseline - a compute "
        "proxy, not a price. optillm reports completion tokens only, so prompt "
        "tokens are uncounted and fan-out approaches are understated: read these "
        "as a floor."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "approach": r.approach,
                    "dataset": r.dataset,
                    "model": r.model,
                    "n": r.n,
                    "accuracy": r.accuracy * 100,
                    "delta vs none (pp)": r.delta_pp,
                    "errors": r.errors,
                    "avg latency (s)": r.avg_latency_s,
                    "avg out tokens": r.avg_completion_tokens,
                    "cost x": r.cost_multiplier,
                }
                for r in rows
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            "cost x": st.column_config.NumberColumn(format="%.2fx"),
        },
    )

    st.markdown("**Accuracy for compute**")
    st.scatter_chart(
        pd.DataFrame(
            [
                {
                    "cost x": r.cost_multiplier or 1.0,
                    "accuracy": r.accuracy * 100,
                    "approach": r.approach,
                }
                for r in rows
            ]
        ),
        x="cost x",
        y="accuracy",
        color="approach",
    )


def render_query_detail(detail: QueryDetail) -> None:
    st.subheader(f"Query {detail.query_id}")
    st.text(detail.question)
    st.markdown(f"**Gold answer:** `{detail.gold}`")
    if detail.any_correct:
        st.markdown(f"**Actual winner:** `{detail.winner}` (cheapest correct approach)")
    else:
        st.markdown(
            f"**Actual winner:** `{detail.winner}` - fallback: no approach was correct"
        )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "approach": a.approach,
                    "": "correct" if a.correct else ("error" if a.error else "wrong"),
                    "predicted": a.predicted,
                    "router picked": a.router_approach or "",
                    "latency (s)": a.latency_s,
                    "tokens": a.total_tokens,
                    "error": a.error or "",
                }
                for a in detail.attempts
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    for attempt in detail.attempts:
        with st.expander(f"Full response - {attempt.approach}"):
            st.text(attempt.response or "(empty)")


def render_router_comparison(report: RouterComparison, scope: str) -> None:
    st.subheader(f"{router_name(report.router)} vs. the actual winner - {scope}")
    if report.n_judged == 0:
        st.info(
            f"No judgeable queries here: the `{report.router}` approach was not "
            f"run, or no approach ever answered correctly."
        )
        return

    left, middle, right = st.columns(3)
    left.metric("Agreement", pct(report.agreement_rate))
    middle.metric("Queries judged", report.n_judged)
    right.metric(
        "No winner",
        report.n_no_winner,
        help="Nothing was correct, so the label fell back to the baseline. These "
        "are excluded from the agreement rate rather than counted as mistakes.",
    )

    st.markdown("**Headroom**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "strategy": "always baseline (`none`)",
                    "accuracy": report.baseline_accuracy * 100,
                    "tokens": report.baseline_tokens,
                },
                {
                    "strategy": f"`{report.router}`",
                    "accuracy": report.router_accuracy * 100,
                    "tokens": report.router_tokens,
                },
                {
                    "strategy": "oracle (always the actual winner)",
                    "accuracy": report.oracle_accuracy * 100,
                    "tokens": report.oracle_tokens,
                },
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"accuracy": st.column_config.NumberColumn(format="%.1f%%")},
    )

    if report.confusions:
        st.markdown("**Most common confusions**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "router picked": c.predicted,
                        "should have won": c.actual,
                        "times": c.count,
                    }
                    for c in report.confusions
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    if report.disagreements:
        st.markdown("**Disagreeing queries**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "query": d.query_id,
                        "dataset": d.dataset,
                        "router picked": d.predicted,
                        "actual winner": d.actual,
                    }
                    for d in report.disagreements
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def render_disagreement_drilldown(
    store: ResultsStore, report: RouterComparison, run_id: str, model: str, key: str
) -> None:
    """Open up one disagreement: what the router picked, what won, and why."""
    if not report.disagreements:
        return

    st.markdown("**Why did it disagree?**")
    choice = pick(
        "Disagreeing query",
        report.disagreements,
        key=key,
        format_func=lambda d: (
            f"{d.dataset} #{d.query_id}: picked {d.predicted}, {d.actual} won"
        ),
    )
    st.caption(
        f"The router picked **{choice.predicted}**; **{choice.actual}** was the "
        f"cheapest approach that answered correctly. Every attempt below is "
        f"what that judgement was made from."
    )
    render_query_detail(
        query_detail(
            store,
            run_id=run_id,
            dataset=choice.dataset,
            model=model,
            query_id=choice.query_id,
        )
    )


# -- app ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    st.set_page_config(page_title="Router lab", layout="wide")
    st.title("Routing / inference-technique results")

    db_path = Path(args.db)
    if not db_path.exists():
        st.error(f"No results database at {db_path}. Sync one down from Supek first.")
        return

    store = open_store(str(db_path), db_path.stat().st_mtime)
    runs = store.list_runs()
    if not runs:
        st.warning(f"{db_path} has no results in it yet.")
        return

    with st.sidebar:
        st.caption(f"Reading `{db_path}` - a snapshot, not a live connection.")
        # Explicit keys: without them Streamlit identifies a widget by its label
        # and options, so the query picker (whose label names the dataset) would
        # be treated as a different widget every time the dataset changed.
        run = st.selectbox(
            "Run",
            runs,
            index=len(runs) - 1,
            format_func=lambda r: f"{r.run_id} ({r.n_results} results)",
            key="run",
        )
        dataset = pick("Dataset", ["all", *run.datasets], key="dataset")
        model = pick("Model", run.models, key="model")

    dataset_filter = None if dataset == "all" else dataset

    leaderboard_tab, query_tab, router_tab = st.tabs(
        ["Leaderboard", "Query drill-down", "Router vs. winner"]
    )

    with leaderboard_tab:
        render_leaderboard(store, run.run_id, dataset_filter, model)

    with query_tab:
        drill_dataset = dataset_filter or run.datasets[0]
        query_ids = store.query_ids(
            run_id=run.run_id, dataset=drill_dataset, model=model
        )
        if not query_ids:
            st.info("No queries for this selection.")
        else:
            query_id = pick(
                f"Query in {drill_dataset}", query_ids, key=f"query-{drill_dataset}"
            )
            render_query_detail(
                query_detail(
                    store,
                    run_id=run.run_id,
                    dataset=drill_dataset,
                    model=model,
                    query_id=query_id,
                )
            )

    with router_tab:
        # Which router is being judged is a choice, not a constant: a sweep can
        # carry the stock router and the finetuned one over identical queries,
        # and the interesting question is how they differ.
        routers = routers_compared(store, run_id=run.run_id, model=model)
        router = pick(
            "Router",
            routers,
            key="router",
            format_func=lambda name: f"{name} - {router_name(name)}",
        )
        if len(routers) == 1:
            st.caption(
                "Only one router ran in this sweep. Sweep `router` and "
                "`router_ft` together to compare them here."
            )

        overall = router_comparison(
            store,
            run_id=run.run_id,
            model=model,
            dataset=dataset_filter,
            router=router,
        )
        render_router_comparison(overall, scope=dataset or "all datasets")
        render_disagreement_drilldown(
            store, overall, run.run_id, model, key=f"disagreement-overall-{router}"
        )

        if dataset_filter is None and len(run.datasets) > 1:
            # Whether the router is good everywhere or only on maths is the
            # question, so category comes before dataset here.
            st.divider()
            st.markdown("### Per category")
            for category, report in comparisons_by_category(
                store, run_id=run.run_id, model=model, router=router
            ).items():
                render_router_comparison(report, scope=category)

            st.divider()
            st.markdown("### Per dataset")
            for name in run.datasets:
                render_router_comparison(
                    router_comparison(
                        store,
                        run_id=run.run_id,
                        model=model,
                        dataset=name,
                        router=router,
                    ),
                    scope=name,
                )


if __name__ == "__main__":
    main()
