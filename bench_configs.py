#!/usr/bin/env python3
"""Benchmark several vLLM *server* configurations back-to-back and emit one
self-contained comparison report.

Where bench_guidellm.py sweeps load against a *single* running server, this
script owns the server lifecycle: for each named config it (re)launches
`vllm serve` with that config's flags, waits for /health, drives the same fixed
GuideLLM load sweep (a shared list of offered request rates, so every config is
measured on the exact same x-axis), samples vLLM's /metrics for KV-cache /
queueing / preemptions, then tears the server down before the next config.

The configs isolate one knob at a time from a shared base so each comparison is
one clean question (see CONFIGS below):
  * batching  -> --max-num-seqs 4 / 32 / 128   (latency vs throughput)
  * memory    -> --gpu-memory-utilization 0.60 vs 0.45   (KV budget under load)
  * kv quant  -> --kv-cache-dtype auto vs fp8   (fp8 halves KV footprint)

Output: bench_results/configs_summary.json  (+ .csv, + comparison .html).
The report is rendered by report_configs.py from that JSON alone, so you can
re-render without re-benchmarking.

Usage:
    # stop any server you have running first (this script binds :8000 itself)
    .venv/bin/python3 bench_configs.py
    .venv/bin/python3 bench_configs.py --dry-run     # print the plan, run nothing
    .venv/bin/python3 bench_configs.py --only seqs-4,fp8-0.45
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import bench_guidellm as bg  # reuse metric parsing / window stats
import report_configs

VLLM_BIN = os.path.expanduser("~/.local/bin/vllm")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
MAX_MODEL_LEN = 4096
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"

# --- the config matrix -----------------------------------------------------
# color_light/dark are validated (dataviz skill) categorical hues; each config
# keeps ONE colour across every chart in the report.
CONFIGS = [
    # batching axis: vary --max-num-seqs at a fixed, comfortable memory budget
    dict(name="seqs-4",   group="batching", label="max-num-seqs = 4",
         gpu_util=0.60, max_num_seqs=4,   kv_dtype="auto",
         color_light="#2a78d6", color_dark="#3987e5",
         blurb="Tiny batch: protects tail latency, caps throughput."),
    dict(name="seqs-32",  group="batching", label="max-num-seqs = 32",
         gpu_util=0.60, max_num_seqs=32,  kv_dtype="auto",
         color_light="#e87ba4", color_dark="#d55181",
         blurb="Balanced batch."),
    dict(name="seqs-128", group="batching", label="max-num-seqs = 128",
         gpu_util=0.60, max_num_seqs=128, kv_dtype="auto",
         color_light="#eda100", color_dark="#c98500",
         blurb="Large batch: maximises throughput, raises latency under load. "
               "Also the 0.60 memory-budget baseline."),
    # memory axis: drop the KV budget, hold batch at 128 so KV can bind
    dict(name="mem-0.45", group="memory", label="gpu-mem-util = 0.45",
         gpu_util=0.45, max_num_seqs=128, kv_dtype="auto",
         color_light="#008300", color_dark="#008300",
         blurb="Tight KV budget (45% of 4 GB) at the same batch cap."),
    # kv-quant axis: fp8 KV cache at the tight budget -> ~2x effective KV
    dict(name="fp8-0.45", group="kvquant", label="fp8 KV @ 0.45",
         gpu_util=0.45, max_num_seqs=128, kv_dtype="fp8",
         color_light="#4a3aa7", color_dark="#9085e9",
         blurb="fp8 KV cache halves bytes/token -> ~2x effective KV at the "
               "same 0.45 budget."),
]


def gpu_mem_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def vllm_version():
    try:
        return subprocess.run([VLLM_BIN, "--version"], capture_output=True,
                              text=True, timeout=15).stdout.strip().splitlines()[-1]
    except Exception:
        return "unknown"


def wait_healthy(timeout: float, log_path: Path, proc) -> bool:
    """Poll /health until 200 or timeout. Returns False on failure/crash."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False  # process exited (OOM / bad flag)
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(cfg: dict, log_path: Path):
    cmd = [VLLM_BIN, "serve", MODEL,
           "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", str(cfg["gpu_util"]),
           "--max-num-seqs", str(cfg["max_num_seqs"]),
           "--port", str(PORT)]
    if cfg["kv_dtype"] != "auto":
        cmd += ["--kv-cache-dtype", cfg["kv_dtype"]]
    # WSL2 has no UVA, so vLLM's newer GPUModelRunnerV2 crashes at init
    # ("RuntimeError: UVA is not available"); force the V1 runner.
    env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)  # own process group
    return proc, " ".join(cmd)


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=10)
    except Exception:
        pass


def wait_gpu_freed(baseline_mib, timeout=60):
    """Wait until GPU memory drops back near the pre-launch baseline so the next
    server doesn't OOM racing the previous one's teardown."""
    if baseline_mib is None:
        time.sleep(6)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = gpu_mem_used_mib()
        if cur is None or cur <= baseline_mib + 300:
            return
        time.sleep(2)


def run_sweep(cfg: dict, args, json_path: Path):
    """Drive the shared multi-rate GuideLLM sweep against the live server while
    sampling /metrics; return parsed per-rate rows."""
    backend = f"kind=openai_http,target={BASE_URL}"
    cmd = [sys.executable, "-m", "guidellm", "run",
           "--backend", backend,
           "--data", f"kind=synthetic_text,prompt_tokens={args.prompt_tokens},"
                     f"output_tokens={args.output_tokens}",
           "--profile", "kind=constant",
           "--override", "profile.rate", args.rates,
           "--constraint", f"kind=max_duration,seconds={args.max_seconds}",
           "--output", f"kind=json,path={json_path}",
           "--disable-progress"]

    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=bg.poll_metrics,
                               args=(BASE_URL, stop_event, samples, args.poll_interval),
                               daemon=True)
    sampler.start()
    subprocess.run(cmd, check=True)
    stop_event.set()
    sampler.join(timeout=2)

    with open(json_path) as f:
        report = json.load(f)

    rows = []
    for b in report["benchmarks"]:
        strat = b["config"]["strategy"]
        m = b["metrics"]
        s, e = b["start_time"], b["end_time"]
        kv_avg, kv_max = bg.window_stats(samples, "kv_cache_usage_perc", s, e)
        _, wait_max = bg.window_stats(samples, "num_requests_waiting", s, e)
        preempt = window_delta(samples, "num_preemptions_total", s, e)
        rows.append({
            "rate": strat.get("rate"),
            "req_s": bg.stat(m, "requests_per_second"),
            "out_tok_s": bg.stat(m, "output_tokens_per_second"),
            "ttft_mean_ms": bg.stat(m, "time_to_first_token_ms"),
            "ttft_p50_ms": bg.pct(m, "time_to_first_token_ms", "p50"),
            "ttft_p95_ms": bg.pct(m, "time_to_first_token_ms", "p95"),
            "ttft_p99_ms": bg.pct(m, "time_to_first_token_ms", "p99"),
            "itl_mean_ms": bg.stat(m, "inter_token_latency_ms"),
            "kv_cache_avg_pct": kv_avg * 100 if kv_avg is not None else None,
            "kv_cache_max_pct": kv_max * 100 if kv_max is not None else None,
            "waiting_max": wait_max,
            "preemptions": preempt,
        })
    rows.sort(key=lambda r: (r["rate"] if r["rate"] is not None else 1e9))
    return rows


def window_delta(samples, key, start, end):
    """Counter increase within [start,end] (last-first), >=0, or None."""
    vals = [s[key] for s in samples if key in s and start <= s["t"] <= end]
    if len(vals) < 1:
        return None
    return max(0.0, vals[-1] - vals[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=320)
    p.add_argument("--rates", default="1,3,5,8,12",
                   help="Shared comma-separated offered request rates (req/s).")
    p.add_argument("--max-seconds", type=int, default=20, help="Duration per rate round.")
    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--startup-timeout", type=float, default=240)
    p.add_argument("--result-dir", default="bench_results")
    p.add_argument("--only", default=None, help="Comma list of config names to run.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    configs = CONFIGS
    if args.only:
        want = {c.strip() for c in args.only.split(",")}
        configs = [c for c in CONFIGS if c["name"] in want]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)

    print(f"Plan: {len(configs)} configs x rates=[{args.rates}] "
          f"prompt/out={args.prompt_tokens}/{args.output_tokens} "
          f"@ {args.max_seconds}s/round on {MODEL}")
    for c in configs:
        print(f"  - {c['name']:9s} util={c['gpu_util']} seqs={c['max_num_seqs']:>3} "
              f"kv={c['kv_dtype']:4s}  [{c['group']}]")
    if args.dry_run:
        return

    baseline_mib = gpu_mem_used_mib()
    vllm_ver = vllm_version()
    print(f"vLLM {vllm_ver} | GPU baseline {baseline_mib} MiB free-of-vLLM\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        log_path = result_dir / f"server_{name}.log"
        json_path = result_dir / f"guidellm_cfg_{name}.json"
        print(f"[{i}/{len(configs)}] === {name} === launching vLLM "
              f"(util={cfg['gpu_util']} seqs={cfg['max_num_seqs']} kv={cfg['kv_dtype']})")
        proc, cmdline = launch_server(cfg, log_path)
        entry = dict(cfg, server_cmd=cmdline, rows=[], status="", peak_mem_mib=None)
        try:
            if not wait_healthy(args.startup_timeout, log_path, proc):
                entry["status"] = "FAILED: server did not become healthy (see log)"
                print(f"    ! {entry['status']}")
                results.append(entry)
                continue
            print("    healthy; running sweep ...")
            entry["peak_mem_mib"] = gpu_mem_used_mib()
            t0 = time.time()
            entry["rows"] = run_sweep(cfg, args, json_path)
            entry["status"] = "ok"
            # refresh peak after load
            m = gpu_mem_used_mib()
            if m is not None:
                entry["peak_mem_mib"] = max(entry["peak_mem_mib"] or 0, m)
            print(f"    done in {time.time()-t0:.0f}s, {len(entry['rows'])} rounds")
        except subprocess.CalledProcessError as e:
            entry["status"] = f"FAILED: guidellm exited {e.returncode}"
            print(f"    ! {entry['status']}")
        finally:
            stop_server(proc)
            wait_gpu_freed(baseline_mib)
        results.append(entry)

    meta = {
        "model": MODEL,
        "gpu": "NVIDIA RTX 3050 (4 GB, WSL2)",
        "vllm_version": vllm_ver,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "rates": args.rates,
        "max_seconds": args.max_seconds,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    summary = {"meta": meta, "configs": results}
    out_json = result_dir / "configs_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # long-format CSV
    csv_path = result_dir / "configs_summary.csv"
    cols = ["config", "group", "rate", "req_s", "out_tok_s", "ttft_p50_ms",
            "ttft_p95_ms", "ttft_p99_ms", "itl_mean_ms", "kv_cache_max_pct",
            "waiting_max", "preemptions"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for c in results:
            for r in c["rows"]:
                f.write(",".join(bg.fmt(r.get(k) if k not in ("config", "group")
                                        else c[{"config": "name", "group": "group"}[k]])
                                 for k in cols) + "\n")

    html_path = result_dir / "configs_comparison.html"
    report_configs.write_comparison_report(summary, html_path)

    print(f"\nSummary JSON: {out_json}")
    print(f"Summary CSV:  {csv_path}")
    print(f"Report:       {html_path}")


if __name__ == "__main__":
    main()
