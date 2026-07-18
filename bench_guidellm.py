#!/usr/bin/env python3
"""Benchmark a running vLLM server with GuideLLM while sampling vLLM's own
Prometheus /metrics endpoint for KV-cache and queueing state.

Neither `guidellm` nor `vllm bench serve` report server-side memory: both only
see client-observed latency/throughput. vLLM exposes the real signal itself at
/metrics (vllm:kv_cache_usage_perc, num_requests_running/waiting,
num_preemptions_total), so this script polls that endpoint on a background
thread while GuideLLM drives load, then joins the two per benchmark round.

GuideLLM's own `--output kind=html` is not used: in 0.7.x its HTML data exporter
emits a shape the pinned frontend bundle can't read (model N/A, 0 benchmarks),
and the page loads its JS live from raw.githubusercontent.com (not offline-safe).
We keep GuideLLM's JSON and render our own self-contained report instead, which
also carries the KV-cache/queue columns GuideLLM's report never had.

Usage:
    vllm serve "Qwen/Qwen2.5-0.5B-Instruct-AWQ" --max-model-len 4096 --gpu-memory-utilization 0.70

    python3 bench_guidellm.py --label baseline --profile sweep --sweep-size 6 --max-seconds 30
"""
import argparse
import html as html_lib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

VLLM_METRICS = {
    "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:num_preemptions_total": "num_preemptions_total",
}
METRIC_LINE_RE = re.compile(r"^(vllm:\w+)\{[^}]*\}\s+([0-9.eE+-]+)\s*$")


def fetch_vllm_metrics(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as resp:
        text = resp.read().decode()
    values = {}
    for line in text.splitlines():
        m = METRIC_LINE_RE.match(line)
        if m and m.group(1) in VLLM_METRICS:
            values[VLLM_METRICS[m.group(1)]] = float(m.group(2))
    return values


def detect_model(base_url: str, fallback: str = None) -> str:
    if fallback:
        return fallback
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as resp:
            data = json.load(resp)
        return data["data"][0]["id"]
    except Exception:
        return "unknown"


def poll_metrics(base_url: str, stop_event: threading.Event, samples: list, interval: float):
    while not stop_event.is_set():
        try:
            values = fetch_vllm_metrics(base_url)
            values["t"] = time.time()
            samples.append(values)
        except Exception:
            pass
        stop_event.wait(interval)


def window_stats(samples: list, key: str, start: float, end: float):
    vals = [s[key] for s in samples if key in s and start <= s["t"] <= end]
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def stat(metrics: dict, name: str, field: str = "mean"):
    try:
        return metrics[name]["successful"][field]
    except (KeyError, TypeError):
        return None


def pct(metrics: dict, name: str, p: str):
    try:
        return metrics[name]["successful"]["percentiles"][p]
    except (KeyError, TypeError):
        return None


def fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def row_label(r: dict) -> str:
    """Short x-axis label per sweep round."""
    s = r["strategy"]
    if s == "synchronous":
        return "sync"
    if s == "throughput":
        return "max"
    if r.get("rate") is not None:
        return f"{r['rate']:.2g}/s"
    return s


# ---------------------------------------------------------------------------
# Self-contained HTML report (combined design: SLO tiles + frontier + dashboard)
# ---------------------------------------------------------------------------

# sequential blue ramp used to encode KV-cache on the frontier markers
KV_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]


def _lerp(v, vmin, vmax, a, b):
    if vmax == vmin:
        return a
    return a + (v - vmin) / (vmax - vmin) * (b - a)


def _load_sorted(rows: list) -> list:
    """Rows ordered by offered load (output throughput asc); rounds with no
    successful throughput sort to the front."""
    return sorted(rows, key=lambda r: (r.get("out_tok/s") or -1.0))


def _derive(rows: list, slo_ms: float) -> dict:
    """Headline numbers + recommendation text from the sweep."""
    valid = [r for r in rows if r.get("out_tok/s")]
    kv_vals = [r["kv_cache_max_%"] for r in rows if r.get("kv_cache_max_%") is not None]
    peak_kv = max(kv_vals) if kv_vals else None
    if not valid:
        return {"peak": None, "safe": None, "peak_kv": peak_kv, "recommendation":
                "No round produced successful throughput — raise --max-seconds or lower load."}
    peak = max(valid, key=lambda r: r["out_tok/s"])
    under = [r for r in valid if r.get("ttft_p99_ms") is not None and r["ttft_p99_ms"] <= slo_ms]
    safe = max(under, key=lambda r: r["out_tok/s"]) if under else None

    kv_note = ""
    if peak_kv is not None:
        if peak_kv < 50:
            kv_note = (f" KV-cache peaks at just {peak_kv:.1f}% — this run is compute-bound, "
                       f"not memory-bound, so there is headroom to raise --max-num-seqs / concurrency "
                       f"or batch size before memory is the limit.")
        else:
            kv_note = (f" KV-cache peaks at {peak_kv:.0f}% — approaching the memory ceiling; "
                       f"KV-cache quantization (fp8) or lower concurrency would buy headroom.")

    if safe is None:
        rec = (f"Every measured round exceeds the {slo_ms:.0f}ms TTFT-p99 SLO. "
               f"Relax the budget or reduce load.{kv_note}")
    elif safe is peak:
        rec = (f"Even at peak load ({peak['out_tok/s']:.0f} tok/s) TTFT-p99 stays under the "
               f"{slo_ms:.0f}ms SLO — you are not latency-limited in this range. Push concurrency "
               f"higher to find the real ceiling.{kv_note}")
    else:
        gain = (peak["out_tok/s"] / safe["out_tok/s"] - 1) * 100
        rec = (f"Run at ≈<b>{safe['req/s']:.1f} req/s</b> — delivers {safe['out_tok/s']:.0f} tok/s "
               f"({safe['out_tok/s'] / peak['out_tok/s'] * 100:.0f}% of peak) while holding TTFT-p99 at "
               f"{safe['ttft_p99_ms']:.0f}ms, under the {slo_ms:.0f}ms SLO. Pushing to peak adds only "
               f"+{gain:.0f}% throughput but raises p99 to {peak['ttft_p99_ms']:.0f}ms and starts "
               f"queueing.{kv_note}")
    return {"peak": peak, "safe": safe, "peak_kv": peak_kv, "recommendation": rec}


def frontier_svg(rows: list, slo_ms: float) -> str:
    """Design A: latency-throughput curve, X=tok/s, Y=TTFT p99, KV as marker shade."""
    pts_data = [(r["out_tok/s"], r["ttft_p99_ms"], r.get("kv_cache_max_%"), row_label(r))
                for r in _load_sorted(rows)
                if r.get("out_tok/s") and r.get("ttft_p99_ms") is not None]
    if len(pts_data) < 2:
        return '<p class="foot">Not enough successful rounds to plot a frontier.</p>'
    W, H = 640, 380
    x0, x1, y0, y1 = 62, 610, 28, 312
    xmax = max(p[0] for p in pts_data) * 1.12
    ymax = max(max(p[1] for p in pts_data), slo_ms) * 1.15
    kvmax = max([p[2] for p in pts_data if p[2] is not None] or [1.0])

    def X(v): return _lerp(v, 0, xmax, x0, x1)
    def Y(v): return _lerp(v, 0, ymax, y1, y0)

    ystep = max(50, round(ymax / 6 / 50) * 50)
    xstep = max(100, round(xmax / 6 / 100) * 100)
    s = [f'<svg viewBox="0 0 {W} {H}">']
    gy = 0
    while gy <= ymax:
        yy = Y(gy)
        s.append(f'<line class="cgrid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>')
        s.append(f'<text class="tick" x="{x0-8:.1f}" y="{yy+3.5:.1f}" text-anchor="end">{gy:.0f}</text>')
        gy += ystep
    gx = 0
    while gx <= xmax:
        s.append(f'<text class="tick" x="{X(gx):.1f}" y="{y1+16:.1f}" text-anchor="middle">{gx:.0f}</text>')
        gx += xstep
    s.append(f'<line class="caxis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    s.append(f'<line class="caxis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    s.append(f'<text class="alab" x="{(x0+x1)/2:.1f}" y="{H-6}" text-anchor="middle">Output throughput (tok/s)  →</text>')
    s.append(f'<text class="alab" transform="translate(14,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">TTFT p99 (ms)  →</text>')
    if slo_ms <= ymax:
        s.append(f'<line class="slo" x1="{x0}" y1="{Y(slo_ms):.1f}" x2="{x1}" y2="{Y(slo_ms):.1f}"/>')
        s.append(f'<text class="slo-lab" x="{x1:.1f}" y="{Y(slo_ms)-5:.1f}" text-anchor="end">SLO — TTFT p99 {slo_ms:.0f}ms</text>')
    pts = [(X(p[0]), Y(p[1])) for p in pts_data]
    d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    s.append(f'<path d="{d}" fill="none" stroke="var(--accent)" stroke-width="2"/>')
    for i, (px, py) in enumerate(pts):
        kv = pts_data[i][2]
        ci = 0 if kv is None else min(len(KV_RAMP) - 1, int(_lerp(kv, 0, kvmax, 0, len(KV_RAMP) - 0.01)))
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{KV_RAMP[ci]}" stroke="var(--surface)" stroke-width="2"/>')
        dy = 17 if i == 0 else -11
        s.append(f'<text class="pt-lab" x="{px:.1f}" y="{py+dy:.1f}" text-anchor="middle">{html_lib.escape(pts_data[i][3])}</text>')
    s.append("</svg>")
    legend = ('<div class="legend"><span>KV-cache peak:</span>'
              f'<span><span class="sw" style="background:{KV_RAMP[0]}"></span>low</span>'
              f'<span><span class="sw" style="background:{KV_RAMP[2]}"></span>mid</span>'
              f'<span><span class="sw" style="background:{KV_RAMP[-1]}"></span>high</span></div>')
    return "".join(s) + legend


def dashboard_svg(rows: list) -> str:
    """Design B: aligned small multiples sharing one offered-load axis."""
    rows = _load_sorted(rows)
    labels = [row_label(r) for r in rows]
    n = len(rows)
    if n == 0:
        return ""
    W = 640
    x0, x1 = 100, 612
    xs = [_lerp(i, 0, max(n - 1, 1), x0, x1) for i in range(n)]
    panels = [
        ("Output throughput", "tok/s", [r.get("out_tok/s") for r in rows], "bar", "var(--accent)"),
        ("TTFT p99", "ms", [r.get("ttft_p99_ms") for r in rows], "line", "var(--accent)"),
        ("KV-cache peak", "%", [r.get("kv_cache_max_%") for r in rows], "area", "var(--accent2)"),
        ("Requests queued", "peak", [r.get("waiting_max") for r in rows], "bar", "var(--accent2)"),
    ]
    ph, gap = 92, 14
    H = len(panels) * (ph + gap) + 34
    s = [f'<svg viewBox="0 0 {W} {H}">']
    for xx in xs:
        s.append(f'<line class="cgrid" x1="{xx:.1f}" y1="10" x2="{xx:.1f}" y2="{H-30:.1f}"/>')
    for pi, (name, unit, vals, kind, color) in enumerate(panels):
        top = 10 + pi * (ph + gap)
        base = top + ph - 4
        nums = [v for v in vals if v is not None]
        vmax = (max(nums) * 1.2) if nums else 1.0
        if vmax == 0:
            vmax = 1.0

        def Y(v): return base - (v / vmax) * (ph - 22)
        s.append(f'<text class="alab" x="{x0-12:.1f}" y="{top+12:.1f}" text-anchor="end">{name}</text>')
        s.append(f'<text class="tick" x="{x0-12:.1f}" y="{top+25:.1f}" text-anchor="end">{unit}</text>')
        s.append(f'<line class="caxis" x1="{x0}" y1="{base:.1f}" x2="{x1}" y2="{base:.1f}"/>')
        if kind == "bar":
            bw = min(30, (x1 - x0) / n * 0.5)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                yy = Y(v)
                s.append(f'<rect x="{xs[i]-bw/2:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{base-yy:.1f}" rx="3" fill="{color}"/>')
                s.append(f'<text class="val" x="{xs[i]:.1f}" y="{yy-4:.1f}" text-anchor="middle">{v:g}</text>')
        else:
            pts = [(xs[i], Y(vals[i])) for i in range(n) if vals[i] is not None]
            if len(pts) >= 2:
                d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
                if kind == "area":
                    da = d + f" L{pts[-1][0]:.1f},{base:.1f} L{pts[0][0]:.1f},{base:.1f} Z"
                    s.append(f'<path d="{da}" fill="{color}" fill-opacity="0.16"/>')
                s.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
            for i in range(n):
                if vals[i] is None:
                    continue
                px, py = xs[i], Y(vals[i])
                s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')
                s.append(f'<text class="val" x="{px:.1f}" y="{py-6:.1f}" text-anchor="middle">{vals[i]:g}</text>')
    for i, xx in enumerate(xs):
        s.append(f'<text class="tick" x="{xx:.1f}" y="{H-14:.1f}" text-anchor="middle">{html_lib.escape(labels[i])}</text>')
    s.append(f'<text class="alab" x="{(x0+x1)/2:.1f}" y="{H-2:.1f}" text-anchor="middle">offered load  →</text>')
    s.append("</svg>")
    return "".join(s)


def _tile(k, v, unit, note, cls):
    return (f'<div class="tile"><div class="k">{html_lib.escape(k)}</div>'
            f'<div class="tval">{v}<span class="u">{html_lib.escape(unit)}</span></div>'
            f'<div class="d {cls}">{html_lib.escape(note)}</div></div>')


def write_html_report(rows: list, headers: list, meta: dict, path: Path, slo_ms: float):
    d = _derive(rows, slo_ms)
    peak, safe, peak_kv = d["peak"], d["safe"], d["peak_kv"]

    if peak is not None:
        peak_under = peak.get("ttft_p99_ms") is not None and peak["ttft_p99_ms"] <= slo_ms
        t_peak = _tile("Peak throughput", f"{peak['out_tok/s']:.0f}", "tok/s",
                       f"at {peak['ttft_p99_ms']:.0f}ms p99" + ("" if peak_under else " — above SLO"),
                       "good" if peak_under else "warn")
    else:
        t_peak = _tile("Peak throughput", "—", "", "no successful rounds", "warn")
    if safe is not None:
        t_safe = _tile(f"Safe throughput @ {slo_ms:.0f}ms", f"{safe['out_tok/s']:.0f}", "tok/s",
                       f"≈ {safe['req/s']:.1f} req/s · {safe['ttft_p99_ms']:.0f}ms p99", "good")
    else:
        t_safe = _tile(f"Safe throughput @ {slo_ms:.0f}ms", "none", "", "all rounds exceed SLO", "crit")
    if peak_kv is not None:
        t_kv = _tile("Peak KV-cache", f"{peak_kv:.1f}", "%",
                     "headroom — not memory-bound" if peak_kv < 50 else "near memory ceiling",
                     "good" if peak_kv < 50 else "warn")
    else:
        t_kv = _tile("Peak KV-cache", "—", "", "not sampled", "warn")

    tiles = f'<div class="tiles">{t_peak}{t_safe}{t_kv}</div>'
    recommendation = f'<div class="read"><b>Recommendation:</b> {d["recommendation"]}</div>'

    # table (sorted by load)
    thead = "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers)
    trows = "".join(
        "<tr>" + "".join(f"<td>{html_lib.escape(fmt(r[h]))}</td>" for h in headers) + "</tr>"
        for r in _load_sorted(rows)
    )
    meta_items = "".join(
        f'<div class="meta-item"><span class="meta-k">{html_lib.escape(k)}</span>'
        f'<span class="meta-v">{html_lib.escape(str(v))}</span></div>'
        for k, v in meta.items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>vLLM benchmark · {html_lib.escape(meta.get('label', ''))}</title>
<style>
:root {{
  color-scheme: light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --accent:#2a78d6; --accent2:#eb6834;
  --good:#0ca30c; --warn:#eda100; --crit:#e34948; --slo:#e34948; --border:rgba(11,11,11,0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --accent:#3987e5; --accent2:#d95926;
    --good:#0ca30c; --warn:#c98500; --crit:#e66767; --slo:#e66767; --border:rgba(255,255,255,0.10);
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:32px; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.45; }}
.wrap {{ max-width:1000px; margin:0 auto; }}
h1 {{ font-size:22px; font-weight:600; margin:0 0 3px; }}
.sub {{ color:var(--ink2); font-size:13px; margin:0 0 22px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 22px; padding:14px 18px; margin-bottom:20px;
  background:var(--surface); border:1px solid var(--border); border-radius:10px; }}
.meta-item {{ display:flex; flex-direction:column; }}
.meta-k {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.meta-v {{ font-size:13px; color:var(--ink); font-variant-numeric:tabular-nums; }}
.tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }}
.tile .k {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.tval {{ font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; margin-top:4px; }}
.tval .u {{ font-size:12px; color:var(--ink2); margin-left:3px; font-weight:400; }}
.tile .d {{ font-size:11.5px; margin-top:4px; }}
.good {{ color:var(--good); }} .warn {{ color:var(--warn); }} .crit {{ color:var(--crit); }}
.read {{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:8px; padding:12px 16px; font-size:13px; color:var(--ink2); margin-bottom:20px; }}
.read b {{ color:var(--ink); }}
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:16px; }}
.card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px; }}
.card .cap {{ font-size:12px; color:var(--muted); margin:0 0 14px; }}
svg {{ width:100%; height:auto; display:block; overflow:visible; }}
.caxis {{ stroke:var(--axis); stroke-width:1; }}
.cgrid {{ stroke:var(--grid); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:10.5px; }}
.alab {{ fill:var(--ink2); font-size:11px; font-weight:500; }}
.val {{ fill:var(--ink2); font-size:10.5px; font-variant-numeric:tabular-nums; }}
.pt-lab {{ fill:var(--ink); font-size:10.5px; font-weight:600; }}
.slo {{ stroke:var(--slo); stroke-width:1.5; stroke-dasharray:4 3; }}
.slo-lab {{ fill:var(--slo); font-size:10.5px; font-weight:600; }}
.legend {{ display:flex; gap:14px; align-items:center; font-size:11px; color:var(--ink2); margin-top:8px; flex-wrap:wrap; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px; vertical-align:-1px; }}
.table-wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ padding:7px 10px; text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums;
  border-bottom:1px solid var(--grid); }}
th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:10px; letter-spacing:.03em; }}
th:first-child, td:first-child {{ text-align:left; }}
tbody tr:last-child td {{ border-bottom:none; }}
.foot {{ color:var(--muted); font-size:11.5px; margin-top:4px; }}
</style></head>
<body><div class="wrap">
<h1>vLLM benchmark — {html_lib.escape(meta.get('label', ''))}</h1>
<p class="sub">GuideLLM load sweep · latency–throughput frontier joined with live vLLM /metrics</p>
<div class="meta">{meta_items}</div>
{tiles}
{recommendation}
<div class="card"><h2>Latency–throughput frontier</h2>
<p class="cap">One curve, idle (sync) → saturated (max). SLO line marks the latency ceiling; marker shade = KV-cache peak.</p>
{frontier_svg(rows, slo_ms)}</div>
<div class="card"><h2>Diagnostics — every metric vs offered load</h2>
<p class="cap">Aligned small multiples: read a vertical slice for the whole system at one load level.</p>
{dashboard_svg(rows)}</div>
<div class="card"><h2>All rounds</h2>
<p class="cap">One benchmark round per row, sorted by offered load. KV-cache &amp; queue sampled from vLLM /metrics.</p>
<div class="table-wrap"><table><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table></div></div>
<p class="foot">Frontier &amp; dashboard extend to multi-config comparison by overlaying one series per config.</p>
</div></body></html>"""
    path.write_text(doc)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Defaults to auto-detect from the server")
    parser.add_argument("--label", default="run", help="Tag for this config; used in output filenames")
    parser.add_argument("--profile", default="sweep",
                         choices=["sweep", "constant", "throughput", "synchronous", "concurrent", "poisson"])
    parser.add_argument("--rate", default=None, help="Comma-separated rate(s); required for constant/poisson")
    parser.add_argument("--sweep-size", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=16,
                        help="Cap concurrent in-flight requests for sweep/throughput/concurrent. "
                             "GuideLLM's default (512) is far too high for a small single-GPU model: "
                             "requests never complete in the window and req/s reports 0.")
    parser.add_argument("--max-seconds", type=int, default=30, help="Max duration per strategy round")
    parser.add_argument("--slo-ttft-ms", type=float, default=200.0,
                        help="TTFT p99 latency budget (ms) for the SLO line and 'safe throughput'")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--result-dir", default="bench_results")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True)
    json_path = result_dir / f"guidellm_{args.label}.json"
    html_path = result_dir / f"guidellm_{args.label}.html"

    model = detect_model(args.base_url, args.model)

    backend = f"kind=openai_http,target={args.base_url}"
    if args.model:
        backend += f",model={args.model}"

    profile = f"kind={args.profile}"
    if args.profile == "sweep":
        profile += f",sweep_size={args.sweep_size},max_concurrency={args.max_concurrency}"
    elif args.profile in ("throughput", "concurrent"):
        profile += f",max_concurrency={args.max_concurrency}"
    elif args.profile in ("constant", "poisson"):
        if not args.rate:
            parser.error(f"--rate is required for --profile {args.profile}")
        profile += f",rate={args.rate}"

    cmd = [
        sys.executable, "-m", "guidellm", "run",
        "--backend", backend,
        "--data", f"kind=synthetic_text,prompt_tokens={args.prompt_tokens},output_tokens={args.output_tokens}",
        "--profile", profile,
        "--constraint", f"kind=max_duration,seconds={args.max_seconds}",
        "--output", f"kind=json,path={json_path}",
    ]

    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=poll_metrics, args=(args.base_url, stop_event, samples, args.poll_interval), daemon=True,
    )
    sampler.start()

    print(f"=== guidellm run: profile={args.profile} label={args.label} model={model} ===")
    subprocess.run(cmd, check=True)

    stop_event.set()
    sampler.join(timeout=2)

    with open(json_path) as f:
        report = json.load(f)

    rows = []
    for b in report["benchmarks"]:
        strategy = b["config"]["strategy"]
        metrics = b["metrics"]
        kv_avg, kv_max = window_stats(samples, "kv_cache_usage_perc", b["start_time"], b["end_time"])
        waiting_avg, waiting_max = window_stats(samples, "num_requests_waiting", b["start_time"], b["end_time"])
        rows.append({
            "strategy": strategy["type_"],
            "rate": strategy.get("rate"),
            "req/s": stat(metrics, "requests_per_second"),
            "out_tok/s": stat(metrics, "output_tokens_per_second"),
            "ttft_mean_ms": stat(metrics, "time_to_first_token_ms"),
            "ttft_p99_ms": pct(metrics, "time_to_first_token_ms", "p99"),
            "itl_mean_ms": stat(metrics, "inter_token_latency_ms"),
            "kv_cache_avg_%": kv_avg * 100 if kv_avg is not None else None,
            "kv_cache_max_%": kv_max * 100 if kv_max is not None else None,
            "waiting_avg": waiting_avg,
            "waiting_max": waiting_max,
        })

    if not rows:
        print("No benchmark rounds in report.")
        return

    headers = list(rows[0].keys())
    print("\n" + " | ".join(f"{h:>14}" for h in headers))
    print("-" * (17 * len(headers)))
    for r in rows:
        print(" | ".join(f"{fmt(r[h]):>14}" for h in headers))

    csv_path = result_dir / f"guidellm_{args.label}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[h]) for h in headers) + "\n")

    meta = {
        "label": args.label,
        "model": model,
        "profile": args.profile,
        "rounds": len(rows),
        "prompt/output tok": f"{args.prompt_tokens}/{args.output_tokens}",
        "max concurrency": args.max_concurrency,
        "sec/round": args.max_seconds,
        "SLO TTFT p99": f"{args.slo_ttft_ms:.0f} ms",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    write_html_report(rows, headers, meta, html_path, args.slo_ttft_ms)

    print(f"\nSaved summary to {csv_path}")
    print(f"Self-contained report: {html_path}")
    print(f"Raw GuideLLM report:   {json_path}")


if __name__ == "__main__":
    main()
