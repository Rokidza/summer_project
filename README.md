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
| [router_lab/policy.py](router_lab/policy.py) | Scoring a routing policy on realised outcomes, against reference policies. |
| [router_lab/grading.py](router_lab/grading.py) | Answer extraction and correctness, one implementation per answer category. |
| [router_lab/datasets.py](router_lab/datasets.py) | Benchmark datasets, declared as config. |
| [router_lab/harness.py](router_lab/harness.py) | Running sweeps against a served model. |
| [router_lab/views.py](router_lab/views.py) | Shaped answers over the store — leaderboards, drill-down, router comparison. |
| [router_lab/dashboard.py](router_lab/dashboard.py) | Streamlit presentation. No SQL, no logic. |
| [router_lab/training/](router_lab/training/) | Finetuning optillm's router on the labels this platform makes. |
| [run_eval.py](run_eval.py) | CLI for a sweep. |
| [train_router.py](train_router.py) | CLI for a finetune. |
| [replay_tracker.py](replay_tracker.py) | CLI to replay a cluster run's JSONL into the local Aim repository. |
| [supek/](supek/) | Apptainer image, model staging, and PBS jobs for the cluster. |

The dependency direction is strict: `dashboard → views → {store, labels}`, and
`training → {store, labels, policy, datasets, grading}` with nothing pointing
back. Nothing outside `store.py` writes SQL, nothing outside `dashboard.py`
imports Streamlit, and nothing outside `training/aim_tracker.py` imports Aim.
Within `training/`, only the modules that actually train import torch — building
examples and scoring outcomes do not.

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

## Finetuning the router

The point of manufacturing labels is to train on them. The finetune continues
optillm's published checkpoint on the winners this platform derived, with only
the classification head and the effort encoder unfrozen — 18k trainable
parameters over a frozen 400M-parameter encoder, which is what makes it fit on
a 4GB laptop GPU.

The deep-learning stack is an optional install group, so the harness and the
dashboard keep working without it:

```bash
uv pip install --python .venv/bin/python3 -e '.[train]'

python train_router.py --db results.sqlite --dry-run   # build examples only
python train_router.py --db results.sqlite --epochs 3
aim up --repo .aim                                     # then look at the run
```

`--dry-run` builds the labelled set, prints its provenance, and scores the
reference policies — no network, no GPU, and no training stack, since all of that
is store, label-rule and dataset work. Do that first: a set built from too few
queries, or one whose labels are all baseline, is a wasted training run.

Four things must agree between training and serving, and each is silent rather
than loud when it does not, so all four are pinned — in
[optillm_router.py](router_lab/training/optillm_router.py) and
[classifier.py](router_lab/training/classifier.py) — and guarded from both
sides: one test reads the upstream plugin's source, another runs a sweep against
the fake server and rebuilds the training text from the messages that actually
went over the wire.

- **The input text** — the system prompt for the dataset's category, then the
  query, in the plugin's exact concatenation.
- **The tokenisation** — same max length, padding and truncation.
- **The effort feature** — the constant the plugin sends at inference. The model
  is never trained under a condition it will not meet.
- **The label space** — optillm's approach list in its original order, so the
  pretrained head is *loaded* rather than re-initialised.

The architecture is vendored here rather than imported from the `optillm/`
checkout: that definition is the contract the checkpoint was serialised
against, and an upstream refactor must not quietly load weights into mismatched
layers.

Splits are ~70/15/15, decided per query from its identity and a recorded seed
and assigned before any two-stage filtering — otherwise a control sample and the
hard queries it was drawn against could straddle the boundary. Each query is
hashed independently rather than the pool being shuffled and sliced, because the
pool grows: a pooled shuffle would reassign existing queries every time a sweep
adds new ones, moving a query out of one finetune's test split and into the next
one's training set. The cost is that the shares are approximate at small
volumes, so the realised counts are recorded rather than assumed.

Provenance (source runs, models actually read, per-dataset counts, coverage,
how many queries no approach solved, seed, label-rule version, content
fingerprint) is logged as run parameters, because metrics can be regenerated by
re-running and provenance cannot. Peak VRAM is recorded too, so "it fits in 4GB"
is a number per run rather than a claim.

### Scoring: what a router would actually have achieved

Agreement with the label is a poor metric. A router that picks a *different*
approach which solved the same query at similar cost scores as a failure while
doing a perfect job, and scores identically to predicting the baseline on a
query only an expensive approach solved — a total failure. So agreement is kept
as a diagnostic (`label_agreement`) and the run is judged on outcomes.

Because the store records, for every (query, approach), whether it was correct
and what it cost, the realised outcome of any routing policy is a table lookup:
no GPU, no serving stack, milliseconds. [policy.py](router_lab/policy.py) takes
a prediction per query and returns **realised accuracy**, **realised tokens**,
**cost as a multiple of the baseline's**, and **coverage** — and it lives with
the results modules, not the training package, because the dashboard is as
legitimate a caller as the training loop. It needs no deep-learning stack.

Two rules make the numbers mean something:

- **Scoring covers the queries with a complete approach matrix.** After a
  two-stage sweep, a query the baseline solved may carry the baseline row alone;
  a policy predicting an unrun approach there has a genuinely *unknown* outcome.
  It is excluded and reported as missing coverage — never imputed, never assumed
  wrong. A policy scoring 0.9 over a fifth of the queries has not been measured,
  and the score says so. Coverage also states *which* of its three causes lost
  it — an unfinished approach matrix, a policy with no opinion, or a prediction
  never run — because those are the sweep's fault, an abstention, and the
  router's reach respectively, and a bare percentage cannot tell them apart.
- **Every policy goes through the same function.** Always-baseline, the best
  single approach, optillm's stock router (scored on the outcome of what it
  picked, not on its own row) and the oracle are scored exactly like the model
  is, over exactly the same queries, and logged at every step so they draw flat
  lines across the run's charts. A realised accuracy of 0.67 means nothing; 0.67
  sitting *below* an always-baseline line at 0.72 is instantly legible as a
  failed run. Since the label rule lets the baseline win every query it answers
  correctly, that line sits high.

`--dry-run` prints the reference scores too, so the bar a finetune has to clear
is visible before a GPU is touched.

Two populations are in play, deliberately: the realised score covers the split's
complete-matrix queries, while the agreement diagnostics below cover every
labelled example, since a label exists whether or not another approach's outcome
on that query does. Coverage is logged beside them so the difference is readable
rather than implied.

Beside the headline numbers, the diagnostics that make a bad run explicable:
macro-F1 (pooled accuracy flatters a majority-class predictor, and the majority
class here is the baseline), realised accuracy per task category, a confusion
matrix per evaluation, and the worst misroutes as inspectable text — query,
prediction, label, and what actually happened, ranked so that throwing away a
win the label's approach achieved outranks any amount of overspending. Picking a
*different* approach that solved the query at no extra cost is not listed at all
— that is the case the realised score exists for, and calling it a misroute would
smuggle label agreement back in through the diagnostics. That last one is the
training-side equivalent of the dashboard's per-query drill-down.

The test split is not scored unless a run asks for it with `--score-test`, and
the headline never silently becomes the test score when it does: a test split
scored on every run of a hyperparameter search is a validation split with extra
steps.

### Recording a cluster run: JSONL, then replay

**Aim's repository lives on local disk and is never committed.** Its backend is
a collection of RocksDB databases, which depend on POSIX locking and mmap
semantics that parallel filesystems handle badly — the same class of hazard the
platform already avoids by packaging dependencies into a container image rather
than unpacking a dependency tree onto Lustre. So it must never be created on
cluster storage.

A cluster-side finetune therefore records to one append-only file on the node's
own disk, and the file is replayed into Aim once it is home:

```bash
# on a compute node
python train_router.py --db results.sqlite --tracker jsonl

# on the laptop, after syncing runs/ back
python replay_tracker.py runs/router-20260730-1200.jsonl --aim-repo .aim
aim up --repo .aim
```

A cluster run and a laptop run then sit side by side in one UI, comparable on
identical metrics, without a tracker reaching across the network from a compute
node. Which one a run uses is configuration and nothing else: the training loop
is written against the `Tracker` protocol, and only
[aim_tracker.py](router_lab/training/aim_tracker.py) imports Aim.

Every call is flushed as it is made, so a job killed at walltime costs the epoch
it was in rather than the run — the closing record is a nicety and replay does
not need it. Replaying the same run twice is *refused*, not repeated: a duplicate
run would be silently averaged into every comparison that included it. Two things
are checked — a receipt written beside the file, and whether the Aim repository
already holds a run replayed from this file's uid, which catches both a fresh
copy synced down from the cluster and a replay that died half way. `--force`
overrides both.

Verified on the laptop by running the same finetune twice under one seed, once
straight into Aim and once via JSONL and replay: identical parameters, metric
series and text panels, differing only in the checkpoint filename (named after
the run) and the provenance replay adds (`replay/source`, `replay/run_uid`).

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
- **Policy scoring** — the same way: outcome tables built by hand, and a
  subprocess check that importing it pulls in no deep-learning stack. Only its
  store wrapper gets a database.
- **Dashboard** — its data-access layer (`views`) against SQLite fixtures. The
  Streamlit rendering layer is thin presentation and is not unit-tested.
- **Training** — a tiny randomly-initialised ModernBERT built from a config in
  the suite ([tests/tiny_router.py](tests/tiny_router.py)), so the loop runs end
  to end with no network and no downloaded weights, and a recording `Tracker`
  ([tests/recording_tracker.py](tests/recording_tracker.py)) the tests assert
  against — never Aim, and never an Aim repository. Replay is tested by feeding
  a JSONL file to that recording tracker and asserting the reconstructed call
  sequence.
- **The upstream contract** — the drift guard reads optillm's plugin as *source*
  rather than importing it, and skips when the checkout is absent.

PBS job scripts, Apptainer builds and cluster process lifecycle are out of
scope for the automated suite — they are verified by the smoke-test checklists
in [supek/README.md](supek/README.md).
