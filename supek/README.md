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
| `sync_results.sh` | Pulls the results database to the laptop. | laptop |

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

## Sizing

`select=1:ngpus=1:ncpus=16:mem=120gb` is the default in both jobs: one A100 for
an 8B AWQ model, cores for optillm's fan-out and the harness's thread pool, and
RAM well clear of the framework's host-side buffers. Always state `mem` — a
`select` without it grants 1800 MiB and the job gets OOM-killed by cgroups.

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
