#!/usr/bin/env python3
"""Sweep vLLM server request rates, correlating latency/throughput with live GPU stats.

Wraps `vllm bench serve` (vLLM's own benchmark client) once per request rate,
while sampling `nvidia-smi` in the background so each row of the summary
table shows how GPU utilization/memory tracked the serving load.

Usage:
    VLLM_USE_V2_MODEL_RUNNER=0 vllm serve "Qwen/Qwen2.5-0.5B-Instruct-AWQ"   --max-model-len 4096   --gpu-memory-utilization 0.70


    python3 bench_sweep.py --model "Qwen/Qwen2.5-0.5B-Instruct-AWQ" --rates 1,5,10,20,50
"""
import argparse
import json
import subprocess
import threading
from pathlib import Path


def sample_gpu(stop_event: threading.Event, samples: list, interval: float = 0.5):
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            util_str, mem_str = out.split(",")
            samples.append((float(util_str), float(mem_str)))
        except Exception:
            pass
        stop_event.wait(interval)


def run_one_rate(base_url, model, rate, num_prompts, input_len, output_len, result_dir):
    result_filename = f"rate_{rate}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", base_url,
        "--endpoint", "/v1/chat/completions",
        "--model", model,
        "--dataset-name", "random",
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(rate),
        "--save-result",
        "--result-dir", str(result_dir),
        "--result-filename", result_filename,
    ]

    gpu_samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop_event, gpu_samples), daemon=True)
    sampler.start()

    print(f"\n=== request-rate={rate} ===")
    subprocess.run(cmd, check=True)

    stop_event.set()
    sampler.join(timeout=2)

    with open(result_dir / result_filename) as f:
        bench_result = json.load(f)

    utils = [u for u, _ in gpu_samples]
    mems = [m for _, m in gpu_samples]
    bench_result["gpu_util_avg"] = sum(utils) / len(utils) if utils else None
    bench_result["gpu_util_max"] = max(utils) if utils else None
    bench_result["gpu_mem_avg_mib"] = sum(mems) / len(mems) if mems else None
    return bench_result


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rates", default="1,5,10,20,50",
                         help="Comma-separated request rates, e.g. '1,5,10,20,inf'")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--result-dir", default="bench_results")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)

    rows = []
    for rate in args.rates.split(","):
        rate = rate.strip()
        try:
            res = run_one_rate(
                args.base_url, args.model, rate, args.num_prompts,
                args.input_len, args.output_len, result_dir,
            )
        except subprocess.CalledProcessError:
            print(f"request-rate={rate} failed, skipping")
            continue
        rows.append({
            "rate": rate,
            "req/s": res.get("request_throughput"),
            "tok/s": res.get("output_throughput"),
            "ttft_mean_ms": res.get("mean_ttft_ms"),
            "ttft_p99_ms": res.get("p99_ttft_ms"),
            "tpot_mean_ms": res.get("mean_tpot_ms"),
            "gpu_util_avg_%": res.get("gpu_util_avg"),
            "gpu_mem_avg_MiB": res.get("gpu_mem_avg_mib"),
        })

    if not rows:
        print("No successful runs.")
        return

    headers = list(rows[0].keys())
    print("\n" + " | ".join(f"{h:>15}" for h in headers))
    print("-" * (18 * len(headers)))
    for r in rows:
        print(" | ".join(f"{fmt(r[h]):>15}" for h in headers))

    csv_path = result_dir / "sweep_summary2.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[h]) for h in headers) + "\n")
    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    main()
