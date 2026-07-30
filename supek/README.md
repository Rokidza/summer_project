# Running the platform on Supek

Supek is SRCE's A100 cluster. Three facts shape everything here:

- The scheduler is **PBS Pro**, not SLURM — `qsub`, `qstat`, `qdel`, `#PBS`.
- The filesystem is **Lustre**, which is punished by many small files — so the
  runtime is an **Apptainer image**, not a pip tree.
- Worker nodes have **no internet** — so weights are staged on a login node and
  jobs run with `HF_HUB_OFFLINE=1`.

> **Status: unverified on the cluster.** Everything in this directory was
> written against the documented Supek environment but has not been run there.
> The checklists below are what to actually execute; expect to adjust the vLLM
> and optillm entrypoint flags to the versions that end up in the image.

## Files

| File | What it does | Where it runs |
|---|---|---|
| `config.sh` | Every knob: model, paths, ports, vLLM defaults. Sourced by the rest. | — |
| `build_image.sh` | Builds the Apptainer `.sif`. | login node |
| `stage_model.sh` | Downloads the model and router classifier into a Lustre `HF_HOME`. | login node |
| `serve.sh` | Starts vLLM + optillm, waits for genuine readiness, tears them down. | inside a job |
| `jobs/interactive.pbs` | Live endpoint to SSH-tunnel into. | `qsub` |
| `jobs/batch_sweep.pbs` | Unattended full sweep, then exit. | `qsub` |
| `jobs/finetune_full.pbs` | Fully-unfrozen router finetune on an A100, then exit. | `qsub` |
| `sync_results.sh` | Pulls the results database, training runs and checkpoints to the laptop. | laptop |

Plus one directory outside this one: `plugins/`, this repo's own optillm plugins
(the finetuned router), discovered by the server from `$OPTILLM_PLUGINS_DIR`.

## Configuration

`config.sh` holds every value, and each one can be overridden from the
environment — including at submission time, which is why no job script hardcodes
a model:

```bash
qsub -v MODEL_REPO=Qwen/Qwen3-14B-AWQ,SERVED_MODEL_NAME=qwen3-14b \
     supek/jobs/batch_sweep.pbs
```

### Why the vLLM defaults look like that

The approaches under test (best-of-N, mixture-of-agents, MCTS) turn one query
into many parallel completions. On a 40GB A100 the KV cache is the scarce
resource and its size is `max_model_len x max_num_seqs`, so the defaults spend
it on **concurrency rather than context**:

| Setting | Default | Why |
|---|---|---|
| `MAX_MODEL_LEN` | 8192 | Over the longest benchmark prompt plus a reasoning budget, well under the model's ceiling. Raise per-run for a dataset that needs it. |
| `MAX_NUM_SEQS` | 256 | Deep enough that a fan-out approach's branches run together instead of queueing. |
| `GPU_MEMORY_UTILIZATION` | 0.90 | AWQ 8B weights are ~6GB, leaving ~30GB for KV cache; the 10% headroom absorbs activation spikes instead of OOM-ing mid-sweep. |
| `TENSOR_PARALLEL_SIZE` | 1 | Must equal `ngpus` in the job's `select`. |

## First-time setup (login node)

```bash
ssh <user>@login-cpu.hpc.srce.hr
cd /lustre/home/$USER
git clone <this repo> router-lab && cd router-lab

./supek/build_image.sh     # several GB; needs internet
./supek/stage_model.sh     # weights + the optillm router classifier
```

Then copy up the optillm checkout, **from the laptop**:

```bash
scp -r ./optillm <user>@login-cpu.hpc.srce.hr:/lustre/home/<user>/optillm
```

The checkout stays a *pristine upstream dependency*: this repo's own plugins are
never copied into it. `serve.sh` passes `--plugins-dir $OPTILLM_PLUGINS_DIR`
(`$PROJECT_DIR/plugins` by default), which is where the finetuned router's plugin
lives — so it is discovered rather than installed, and an upstream refresh cannot
lose it.

This step is not optional and not a convenience. optillm is *not* installed
into the image, and the jobs run it from this checkout, because the copy this
project uses carries local changes a released version does not have:

- the router's device is configurable and defaults to GPU (issue #8);
- responses carry `optillm_router_approach`, which is the only way the harness
  learns what the router picked — without it the whole router-vs-winner view
  (issue #12) is silently empty on every cluster result.

`serve.sh` checks for the checkout and refuses to start without it, rather than
serving a stock optillm that reports nothing.

`stage_model.sh` finishes by loading the staged model with `HF_HUB_OFFLINE=1`,
which is the real proof that a worker node can use it.

**Checklist — issue #6**

- [ ] `apptainer build` completes and `$SIF` exists.
- [ ] The optillm checkout is at `$OPTILLM_DIR` and `serve.sh` no longer refuses.
- [ ] `logs/<jobid>/optillm.log` shows it started from `$OPTILLM_DIR/optillm.py`,
      not from a site-packages copy.
- [ ] On a GPU node: `apptainer test --nv $SIF` reports vLLM, optillm's deps and
      a visible GPU. Get one with
      `qsub -I -q gpu-test -l select=1:ngpus=1:ncpus=8:mem=64gb -l walltime=00:30:00`.
- [ ] `du -sh $MODEL_DIR` looks like a complete model, with `.safetensors` present.
- [ ] The offline load at the end of `stage_model.sh` succeeds.
- [ ] No pip/conda tree was created on Lustre for the runtime.

## Interactive session

```bash
qsub supek/jobs/interactive.pbs
qstat -u $USER                       # wait for R
cat router_lab_interactive.o<jobid>  # prints the exact tunnel command
```

Then from the laptop:

```bash
ssh -N -L 8000:<worker-node>:8000 <user>@login-cpu.hpc.srce.hr
curl http://localhost:8000/v1/models

python run_eval.py --model qwen3-8b \
    --base-url http://localhost:8000/v1 \
    --approaches none bon router --dataset gsm8k --limit 20 \
    --db results.sqlite
```

Nothing about the harness changes between a laptop-served model and this — only
`--base-url`.

**Checklist — issues #7 and #9**

- [ ] `/v1/models` reports `$SERVED_MODEL_NAME` (the job asserts this itself).
- [ ] A chat completion returns a sensible answer (the job smoke-tests this).
- [ ] `{"optillm_approach": "bon"}` is honoured, and the `router` approach comes
      back with `optillm_router_approach` naming what it picked.
- [ ] Concurrent requests are served in parallel, not visibly serialised —
      e.g. 32 at once should not take 32x one.
- [ ] Memory holds under sustained concurrency; `nvidia-smi` on the node shows
      no OOM and `logs/<jobid>/vllm.log` shows no preemption storm.
- [ ] A small sweep's results appear in the dashboard.
- [ ] The job ends itself at walltime.

**Checklist — issue #20** (the finetuned router as an approach)

Submit with a checkpoint and it becomes an ordinary approach:

```bash
qsub -v ROUTER_FT_CHECKPOINT=/lustre/home/$USER/checkpoints/router-<run>.pt,\
APPROACHES="none bon moa router router_ft" supek/jobs/batch_sweep.pbs
```

- [ ] `logs/<jobid>/optillm.log` shows `Loaded local plugin: router_ft` from
      `$OPTILLM_PLUGINS_DIR/optillm/plugins`, and the upstream checkout at
      `$OPTILLM_DIR` is unmodified (`git -C $OPTILLM_DIR status` is clean, if it
      is a checkout at all).
- [ ] `{"optillm_approach": "router_ft"}` answers, and the response's
      `optillm_router_approach` names an approach — the log line
      `Finetuned router predicted approach: ...` agrees with it.
- [ ] `Loading finetuned router <path> on cuda` appears once, not per request.
- [ ] Pointing `ROUTER_FT_CHECKPOINT` at a different checkpoint changes what is
      served, with no rebuild of the image.
- [ ] Unset, the job says the approach will refuse, and a `router_ft` request
      fails with the variable named rather than silently routing to the baseline.
- [ ] A sweep with both `router` and `router_ft` records both, over the same
      queries, and the dashboard's router picker compares either.

**Checklist — issue #8** (router device)

- [ ] With `OPTILLM_ROUTER_DEVICE` unset, `logs/<jobid>/optillm.log` prints
      `Loading router classifier on cuda`.
- [ ] `nvidia-smi` shows the classifier's ~1.6GB alongside vLLM.
- [ ] `OPTILLM_ROUTER_DEVICE=cpu` still works (already verified on the laptop).
- [ ] Concurrent `router` requests all report a predicted approach.

## Unattended sweep

```bash
qsub supek/jobs/batch_sweep.pbs
# or narrow it:
qsub -v DATASETS="gsm8k boolq",APPROACHES="none bon moa router",LIMIT=50 \
     supek/jobs/batch_sweep.pbs
```

It starts the stack, waits for both servers to genuinely answer, runs every
approach across every dataset under one run id, stops the servers, and exits.
A dataset that fails is logged and the sweep continues, but the job exits
non-zero so the failure cannot pass unnoticed.

Then, on the laptop:

```bash
./supek/sync_results.sh
streamlit run router_lab/dashboard.py -- --db results.sqlite
```

**Checklist — issue #11**

- [ ] One `qsub` runs the whole sweep with no intervention.
- [ ] The sweep does not start before both servers answer HTTP.
- [ ] `-v DATASETS=...,APPROACHES=...` changes what runs.
- [ ] `$RESULTS_DB` contains the sweep as a single run.
- [ ] Both servers are gone and the job exits cleanly at the end.
- [ ] Killing vLLM mid-job fails the sweep loudly with its log quoted, rather
      than hanging to walltime.
- [ ] The synced database opens in the dashboard showing all approaches and
      datasets with cost multipliers against the baseline.

## Fully-unfrozen router finetune

The laptop's 4GB GPU can run `--regime head` (and, tightly, `top_layers`), but
not `--regime full`: fully finetuning the ~400M-parameter encoder needs roughly
6GB for weights, gradients and optimizer state before activations even enter
the picture. This job is the escape hatch, not the default loop - the other two
regimes still run on the laptop with `train_router.py` directly.

```bash
qsub supek/jobs/finetune_full.pbs
# or override:
qsub -v EPOCHS=5,LEARNING_RATE=1e-5,RUN_NAME=full-2 supek/jobs/finetune_full.pbs
```

It reads `$RESULTS_DB` from Lustre, trains inside the existing image (torch,
transformers and safetensors are already baked in - no pip/conda tree is
created on Lustre for this), and records through the **JSONL tracker**, never
Aim: an Aim repository must never be created on cluster storage. Metrics and
the checkpoint land in `$TRAINING_RUNS_DIR` and `$CHECKPOINT_DIR`, which is
where the existing sync workflow now looks.

Then, on the laptop:

```bash
./supek/sync_results.sh
python replay_tracker.py runs/router-<run-name>.jsonl
aim up --repo .aim
```

The replayed run sits beside laptop-trained `head`/`top_layers` runs in the same
Aim UI, comparable on the same charts - the whole point of the JSONL tracker.
The checkpoint it produced is loadable by the `router_ft` serving plugin with
no modification:

```bash
qsub -v ROUTER_FT_CHECKPOINT=$CHECKPOINT_DIR/router-<run-name>.pt,\
APPROACHES="none bon moa router router_ft" supek/jobs/batch_sweep.pbs
```

**Checklist — issue #21**

- [ ] The job runs the fully-unfrozen regime and exits on its own, with no
      idle GPU time - `logs/<jobid>/train.log` shows `regime full`.
- [ ] It records through the JSONL tracker; `apptainer` never has `aim`
      importable inside it, and no `.aim` directory appears on Lustre.
- [ ] `$TRACKER_FILE` and the checkpoint under `$CHECKPOINT_DIR` are present
      after the job ends.
- [ ] `./supek/sync_results.sh` pulls both down into `./runs` and
      `./checkpoints` with no manual `scp`.
- [ ] `python replay_tracker.py runs/router-<run-name>.jsonl` puts the run into
      the local Aim repo, with the same parameters and metrics the log printed,
      appearing alongside laptop runs.
- [ ] `ROUTER_FT_CHECKPOINT` pointed at the pulled-down checkpoint serves it
      with no code change and no rebuild.
- [ ] No bare pip or conda tree was created on Lustre for training dependencies.

## Sizing

`select=1:ngpus=1:ncpus=16:mem=120gb` is the default in the two serving jobs
(`interactive.pbs`, `batch_sweep.pbs`): one A100 for an 8B AWQ model, cores for
optillm's fan-out and the harness's thread pool, and RAM well clear of the
framework's host-side buffers. Always state `mem` — a `select` without it
grants 1800 MiB and the job gets OOM-killed by cgroups.

`finetune_full.pbs` asks for less — `ngpus=1:ncpus=8:mem=64gb` — because it
never starts vLLM or optillm: there is no fan-out to give cores to, and the
~400M-parameter encoder plus optimizer state needs a fraction of the serving
jobs' memory.

Scaling to a bigger model means more GPUs *and* matching `TENSOR_PARALLEL_SIZE`:

```bash
qsub -l select=1:ngpus=4:ncpus=32:mem=240gb \
     -v MODEL_REPO=Qwen/Qwen3-32B-AWQ,SERVED_MODEL_NAME=qwen3-32b,TENSOR_PARALLEL_SIZE=4 \
     supek/jobs/batch_sweep.pbs
```

## When something breaks

| Symptom | Cause |
|---|---|
| Download fails inside a job | Worker nodes have no internet. Re-run `stage_model.sh` on a login node. |
| `router` requests all fall back | The classifier was not staged. `stage_model.sh` fetches it; without it the first routed request tries the hub and fails. |
| OOM-killed | `mem` too low, or `GPU_MEMORY_UTILIZATION` too high for the model. |
| GPU not found | Missing `--nv`, wrong queue, or no `ngpus` in `select`. |
| File missing in the job, present on login | It is on `/storage` (Štampar), invisible to workers. Move it to `/lustre`. |
| Server unreachable from the laptop | Tunnel through a login node to `<worker>:<port>`, not to the login node itself. |
