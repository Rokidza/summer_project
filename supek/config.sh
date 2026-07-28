# Shared configuration for every Supek script and job in this directory.
#
# Sourced, never executed. Every value can be overridden from the environment,
# so a job can be submitted against a different model or serving config without
# editing a file:
#
#     qsub -v MODEL_REPO=Qwen/Qwen3-14B-AWQ,SERVED_MODEL_NAME=qwen3-14b jobs/batch_sweep.pbs
#
# shellcheck shell=bash

# -- what to serve --------------------------------------------------------
# The model is a config value, not a hardcode: these two lines are the only
# place its identity lives. MODEL_REPO is the HuggingFace repo staged by
# stage_model.sh; SERVED_MODEL_NAME is the name clients (and --model on the
# eval harness) use.
: "${MODEL_REPO:=Qwen/Qwen3-8B-AWQ}"
: "${SERVED_MODEL_NAME:=qwen3-8b}"

# -- where things live ----------------------------------------------------
# Everything a job reads must be on Lustre; worker nodes cannot see /storage.
: "${LUSTRE_HOME:=/lustre/home/$USER}"
: "${HF_HOME:=$LUSTRE_HOME/hf}"
: "${MODEL_DIR:=$HF_HOME/models/$(basename "$MODEL_REPO")}"
: "${SIF:=$LUSTRE_HOME/router-lab.sif}"
: "${PROJECT_DIR:=$LUSTRE_HOME/router-lab}"
: "${RESULTS_DB:=$LUSTRE_HOME/results/results.sqlite}"

# optillm runs from a staged source checkout rather than the image's PyPI
# copy, because this project's optillm carries local changes a released
# version does not have: the router's configurable device, and the
# optillm_router_approach field that the harness reads to learn what the
# router picked. See supek/README.md for how to stage it.
: "${OPTILLM_DIR:=$LUSTRE_HOME/optillm}"

# Where to SSH in, and the tunnel target. Kept here so no script hardcodes it.
: "${SUPEK_LOGIN:=login-cpu.hpc.srce.hr}"

# -- ports ----------------------------------------------------------------
# vLLM holds the weights; optillm proxies in front of it. Clients and the eval
# harness only ever talk to optillm.
: "${VLLM_PORT:=8001}"
: "${OPTILLM_PORT:=8000}"

# Both servers must bind the node's external interface, not loopback: an SSH
# tunnel is opened from the laptop *through a login node* to <worker>:<port>,
# and the login node cannot reach a compute node's 127.0.0.1. Compute nodes sit
# on the cluster's private network, so this exposes nothing to the internet.
: "${BIND_HOST:=0.0.0.0}"

# -- vLLM serving defaults ------------------------------------------------
# Deliberately biased toward *concurrency* rather than context length.
#
# The approaches under test (best-of-N, mixture-of-agents, MCTS) fan a single
# query out into many parallel completions. On a 40GB A100 the KV cache is the
# scarce resource, and its size is (max_model_len x max_num_seqs): spending it
# on context length nobody uses would serialise exactly the workloads this
# platform exists to measure.
#
#   MAX_MODEL_LEN 8192   - comfortably over the longest benchmark prompt plus a
#                          reasoning budget, and far below the model's ceiling.
#                          Raise it per-run for a dataset that genuinely needs it.
#   MAX_NUM_SEQS 256     - deep enough that a fan-out approach's branches run
#                          together rather than queueing behind each other.
#   GPU_MEMORY_UTILIZATION 0.90
#                        - AWQ-quantised 8B weights are ~6GB, leaving roughly
#                          30GB of the 36GB budget for KV cache. The 10% head-
#                          room absorbs activation spikes under full batches
#                          instead of OOM-ing the server mid-sweep.
#   TENSOR_PARALLEL_SIZE 1
#                        - must equal the ngpus in the job's select statement.
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=256}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${TENSOR_PARALLEL_SIZE:=1}"

# -- optillm --------------------------------------------------------------
# The pretrained router's classifier device. "auto" takes the GPU when one is
# present (Supek) and falls back to CPU (laptop) - see router_plugin.py.
: "${OPTILLM_ROUTER_DEVICE:=auto}"

# -- container bind paths -------------------------------------------------
# Apptainer only sees the host paths it is told to bind. Binding the Lustre
# home covers the image's three inputs at once: the staged weights under
# HF_HOME, the project checkout, and the results database. $APPTAINER_BIND is
# read by every `apptainer exec`, so no call site has to repeat it.
: "${APPTAINER_BIND:=$LUSTRE_HOME}"
export APPTAINER_BIND

# -- offline mode ---------------------------------------------------------
# Worker nodes have no internet. Everything must already be in HF_HOME.
# Apptainer passes the host environment into the container by default, so
# exporting these here is what makes them apply *inside* the image too.
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPTILLM_ROUTER_DEVICE
