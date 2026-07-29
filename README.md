# Router lab

A research platform for answering one question: **which LLM inference-time
technique works best for which kind of query — and is optillm's pretrained
router any good at picking?**

It serves a model with vLLM, puts optillm in front of it, runs benchmark
datasets through each of optillm's approaches, records every result in a
queryable SQLite store, derives which approach *actually* won per query, and
shows the lot in a local dashboard.

```
  vLLM  ──  holds the weights
    ▲
    │  OpenAI API
  optillm  ──  picks the inference-time approach (bon, moa, mcts, router, …)
    ▲
    │  OpenAI API (HTTP only — the harness never imports optillm)
  run_eval.py  ──►  results.sqlite  ──►  dashboard
```

## Layout

| Path | What it is |
|---|---|
| [router_lab/store.py](router_lab/store.py) | The SQLite results store. The **only** module that knows the schema. |
| [router_lab/labels.py](router_lab/labels.py) | The label rule: which approach actually won, per query. |
| [router_lab/grading.py](router_lab/grading.py) | Answer extraction and correctness, one implementation per answer category. |
| [router_lab/datasets.py](router_lab/datasets.py) | Benchmark datasets, declared as config. |
| [router_lab/harness.py](router_lab/harness.py) | Running sweeps against a served model. |
| [router_lab/views.py](router_lab/views.py) | Shaped answers over the store — leaderboards, drill-down, router comparison. |
| [router_lab/dashboard.py](router_lab/dashboard.py) | Streamlit presentation. No SQL, no logic. |
| [run_eval.py](run_eval.py) | CLI for a sweep. |
| [supek/](supek/) | Apptainer image, model staging, and PBS jobs for the cluster. |

The dependency direction is strict: `dashboard → views → {store, labels}`.
Nothing outside `store.py` writes SQL, and nothing outside `dashboard.py`
imports Streamlit.

## Running a sweep

With vLLM on `:8001` and optillm proxying on `:8000`:

```bash
python run_eval.py --model qwen3-8b \
    --approaches none bon moa router \
    --dataset gsm8k --limit 50 \
    --db results.sqlite
```

Every labelled query costs one run of each approach, and most approaches fan
out into many LLM calls — so a full sweep is the platform's dominant expense.
Most of it is spent on queries the baseline already answers correctly, which
the label rule hands to the baseline regardless. `--two-stage` runs the
baseline over the whole pool first, then the other approaches over the queries
it got *wrong*, plus a seeded control sample of the ones it got right (the
control is what keeps dataset-level accuracy and cost comparisons honest):

```bash
python run_eval.py --model qwen3-8b \
    --approaches none bon moa router \
    --dataset gsm8k --limit 500 \
    --two-stage --control-fraction 0.1 --seed 0
```

Both stages record under one run id, and the sweep's configuration is stored
alongside its results, so how a run was produced is readable back from the
database. Which approaches a given query was actually run through is *derived*
from the rows (`ResultsStore.approach_coverage`), never written down.

Results accumulate in one database across runs, datasets, approaches and
models. Browse them:

```bash
streamlit run router_lab/dashboard.py -- --db results.sqlite
```

Then open the URL it prints. It will not open a browser for you: under WSL with
no `xdg-open`/`wslview`/`$BROWSER`, Streamlit's browser launch blocks the server
before it answers anything — the port listens and every request hangs. So
`.streamlit/config.toml` forces `headless = true`.

The dashboard is a passive viewer over the file — it never connects to a
cluster. On Supek, sweeps run unattended and the database is pulled down
afterwards; see [supek/README.md](supek/README.md).

## Concepts

**Cost is never money.** It is completion tokens, total tokens, latency and
call count, reported as a multiplier against the `none` baseline. A dollar
figure would be fictional on your own GPU.

Two caveats, both making cost an *under*-estimate: optillm's usage block
reports completion tokens only (so prompt tokens, paid once per fan-out
branch, go uncounted), and it reports no LLM call count (so `call_count` is 1
per request). The harness reads both from the response the moment optillm
starts reporting them — until then, read cost multipliers as a floor.

**The label rule.** For each query, the winner is the cheapest approach that
answered correctly; if none did, it is the baseline. `router` is excluded from
winning — it chooses a technique rather than being one, so letting it win its
own comparison would be circular.

**Approach vetting order is benchmark first, audit second.** Run everything,
then dig into whatever looks anomalous.

## Adding a dataset

Add an entry to `DATASETS` in [router_lab/datasets.py](router_lab/datasets.py).
If its answers fit an existing category (`numeric`, `boolean`,
`multiple_choice`, `code`), that is the whole change — no new grading code:

```python
"my_bench": {
    "path": "org/my-bench",
    "name": None,
    "split": "test",
    "category": "multiple_choice",
    "question_template": "{question}\n\n{choices}",
    "list_style": {"choices": "lettered"},
    "answer_key": "answer",
},
```

Two optional keys cover datasets that nest their options. A template reaches
into a nested field with `{choices[text]}`, and `label_path` names the list of
the options' own labels — options are always lettered by position, so where a
dataset labels them itself (ARC-Challenge labels some rows `1`-`4`) the gold is
that label's *position*, not its face value:

```python
"arc_challenge": {
    ...
    "question_template": "{question}\n\n{choices[text]}",
    "list_style": {"choices": "lettered"},
    "answer_key": "answerKey",
    "label_path": "choices.label",
},
```

A genuinely new answer shape adds one entry to each of `SYSTEM_PROMPTS`,
`EXTRACTORS` and `NORMALIZERS` in
[router_lab/grading.py](router_lab/grading.py) — and then the next dataset of
that shape is config-only again.

## Tests

```bash
python -m pytest
```

Tests sit at the seams and use real-but-cheap dependencies rather than mocks:

- **Store** — a real temp-file SQLite database, never a mocked DB.
- **Harness** — a real HTTP server speaking the OpenAI API
  ([tests/fake_server.py](tests/fake_server.py)), never a real vLLM+optillm.
- **Label rule** — a pure function; constructed result sets, no I/O.
- **Dashboard** — its data-access layer (`views`) against SQLite fixtures. The
  Streamlit rendering layer is thin presentation and is not unit-tested.

PBS job scripts, Apptainer builds and cluster process lifecycle are out of
scope for the automated suite — they are verified by the smoke-test checklists
in [supek/README.md](supek/README.md).
