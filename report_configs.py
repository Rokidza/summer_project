#!/usr/bin/env python3
"""Render the vLLM comparison report from configs_summary.json.

This is written as a *teaching page*, not a dashboard: it explains what LLM
inference, latency and memory are to someone new to the field, and uses our
measured experiments (with the exact vLLM commands) as the worked examples.

Kept separate from the benchmark runner so it can be regenerated from saved
data alone:

    .venv/bin/python3 report_configs.py bench_results/configs_summary.json

Charts follow the dataviz skill: one chart = one question; a shared offered-load
x-axis so configs overlay honestly; validated colourblind-safe hues, one per
config, held constant across every panel and theme (light/dark); TTFT drawn as a
p50->p99 band so variance is visible; legend + direct labels; crosshair+tooltip.
"""
import html as html_lib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# panel metadata: title, unit, data key, and a one-line plain-English meaning
# ---------------------------------------------------------------------------
PANELS = {
    "throughput": dict(title="Output throughput", unit="tok/s", key="out_tok_s", band=False,
        explain="Total tokens the server emits every second across all users at once — its "
                "aggregate work rate. Higher is better."),
    "ttft": dict(title="Time to first token (p50–p99)", unit="ms", key="ttft_p99_ms", band=True,
        explain="How long a user waits after sending a request before the very first token "
                "appears. The shaded band spans typical (p50) to near-worst-case (p99). Lower is better."),
    "itl": dict(title="Inter-token latency", unit="ms/tok", key="itl_mean_ms", band=False,
        explain="Once generation starts, the gap between streamed tokens — how fast the text "
                "flows out. Lower is better."),
    "kv": dict(title="Peak KV-cache usage", unit="% of budget", key="kv_cache_max_pct", band=False,
        explain="How full the KV-cache memory pool got at its busiest, as a share of the budget "
                "vLLM reserved for it. Approaching 100% means the server starts evicting work."),
    "preempt": dict(title="Preemptions", unit="count / round", key="preemptions", band=False,
        explain="How many times vLLM had to kick a running request out of GPU memory because "
                "the KV-cache filled up. Zero means memory was never the bottleneck."),
}

# ---- experiment sections, each a self-contained lesson --------------------
SECTIONS = [
    dict(id="batching", num=1, knob="--max-num-seqs",
         title="Experiment 1 — Batching: trading latency for throughput",
         highlight="--max-num-seqs",
         what="A GPU is far more efficient when it works on many sequences at once. vLLM does this "
              "with <b>continuous batching</b>: on every decode step it advances a whole batch of "
              "in-flight requests together, filling the GPU. The flag "
              "<code>--max-num-seqs</code> caps how many requests may run in that batch at one time.",
         why="A bigger cap means more tokens computed per GPU step, so higher total throughput. But "
              "each step also gets heavier — every user waits for the whole batch to advance — and if "
              "the cap is <em>too small</em>, arriving requests pile up in a queue instead of being "
              "served. That is why this single number pushes latency and throughput in opposite "
              "directions. It is the first knob to reach for.",
         configs=["seqs-4", "seqs-32", "seqs-128"], panels=["throughput", "ttft", "itl"]),
    dict(id="memory", num=2, knob="--gpu-memory-utilization",
         title="Experiment 2 — Memory: how much room the KV-cache gets",
         highlight="--gpu-memory-utilization",
         what="After the model weights are loaded, whatever GPU memory is left becomes the "
              "<b>KV-cache pool</b> — the scratch space that holds every in-flight request's attention "
              "state (see the primer above). <code>--gpu-memory-utilization</code> sets the fraction of "
              "the card vLLM is allowed to claim; a bigger fraction leaves more room for the KV-cache, "
              "so more requests (or longer prompts) can run before it fills.",
         why="If the KV-cache fills, vLLM must <b>preempt</b> — evict a running request and recompute it "
              "later — which spikes latency and drops throughput. So the question is: on this hardware and "
              "workload, does a tighter memory budget actually hurt, or is something else the limit first?",
         configs=["seqs-128", "mem-0.45"], panels=["throughput", "kv", "preempt"]),
    dict(id="kvquant", num=3, knob="--kv-cache-dtype fp8",
         title="Experiment 3 — Quantization: the same cache in half the bytes",
         highlight="--kv-cache-dtype",
         what="Every token in the KV-cache is normally stored as 16-bit numbers. <b>KV-cache "
              "quantization</b> stores them as 8-bit floats instead (<code>--kv-cache-dtype fp8</code>), "
              "halving the bytes per token for a tiny, usually invisible, accuracy cost.",
         why="Half the bytes per token means roughly <b>twice</b> the effective KV-cache — you can hold "
              "twice the concurrency or twice the context in the same memory budget. Here we hold the "
              "budget fixed at the tight 0.45 setting and switch fp8 on, to isolate exactly what it buys.",
         configs=["mem-0.45", "fp8-0.45"], panels=["kv", "throughput", "ttft"]),
]


def _cls(name):
    return "c-" + name.replace(".", "-")


def _by_name(summary):
    return {c["name"]: c for c in summary["configs"]}


def _rows_ok(cfg):
    return cfg.get("status") == "ok" and cfg.get("rows")


def _xy(cfg, key):
    out = []
    for r in cfg.get("rows", []):
        if r.get("rate") is not None and r.get(key) is not None:
            out.append((float(r["rate"]), float(r[key])))
    return sorted(out)


def _fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:,.{nd}f}"


def _peak(cfg, key):
    return max((y for _, y in _xy(cfg, key)), default=None)


def _at_max_rate(cfg, key):
    pts = _xy(cfg, key)
    return pts[-1][1] if pts else None


def _at_rate(cfg, key, rate):
    for r in cfg.get("rows", []):
        if r.get("rate") == rate:
            return r.get(key)
    return None


def _matched(a, b, key):
    da = {r["rate"]: r.get(key) for r in a.get("rows", [])}
    db = {r["rate"]: r.get(key) for r in b.get("rows", [])}
    out = []
    for rt in sorted(set(da) & set(db)):
        if da[rt] is not None and db[rt] is not None:
            out.append((rt, da[rt], db[rt]))
    return out


def _fp8_kv_ratio(a, b):
    m = [(rt, va, vb) for rt, va, vb in _matched(a, b, "kv_cache_max_pct") if va and va > 1]
    if not m:
        return None
    rt, va, vb = m[len(m) // 2] if len(m) < 4 else m[-2]
    return (vb / va * 100, rt, va, vb)


# ---------------------------------------------------------------------------
# line-chart panel  (returns svg_html, hover_blob)
# ---------------------------------------------------------------------------
def line_panel(spec, cfgs, all_rates):
    W, H = 560, 300
    x0, x1, y0, y1 = 58, 470, 22, 250
    key, band = spec["key"], spec["band"]

    series, ymax = [], 0.0
    for cfg in cfgs:
        pts = _xy(cfg, key)
        if pts:
            ymax = max(ymax, max(p[1] for p in pts))
        band_pts = None
        if band:
            bp = [(float(r["rate"]), r.get("ttft_p50_ms"), r.get("ttft_p99_ms"))
                  for r in cfg.get("rows", []) if r.get("rate") is not None]
            band_pts = [(x, lo, hi) for x, lo, hi in bp if lo is not None and hi is not None]
            if band_pts:
                ymax = max(ymax, max(hi for _, _, hi in band_pts))
        series.append((cfg, pts, band_pts))
    ymax = (ymax or 1.0) * 1.18
    xmin, xmax = 0.0, max(all_rates) * 1.06
    ystep = _nice_step(ymax / 5)
    ytop = ystep * (int(ymax / ystep) + 1)

    def X(v): return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
    def Y(v): return y1 - (v / ytop) * (y1 - y0)

    s = [f'<svg viewBox="0 0 {W} {H}" class="lc" role="img" '
         f'aria-label="{html_lib.escape(spec["title"])} versus offered request rate">']
    gy = 0.0
    while gy <= ytop + 1e-6:
        yy = Y(gy)
        s.append(f'<line class="cgrid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>')
        s.append(f'<text class="tick" x="{x0-8:.1f}" y="{yy+3.5:.1f}" text-anchor="end">{_tick(gy)}</text>')
        gy += ystep
    for rv in all_rates:
        s.append(f'<text class="tick" x="{X(rv):.1f}" y="{y1+16:.1f}" text-anchor="middle">{_tick(rv)}</text>')
    s.append(f'<line class="caxis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    s.append(f'<line class="caxis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    s.append(f'<text class="alab" x="{(x0+x1)/2:.1f}" y="{H-24:.1f}" text-anchor="middle">'
             f'Offered load — requests arriving per second  →</text>')
    s.append(f'<text class="alab" transform="translate(13,{(y0+y1)/2:.1f}) rotate(-90)" '
             f'text-anchor="middle">{html_lib.escape(spec["title"])} ({html_lib.escape(spec["unit"])})</text>')

    for cfg, pts, band_pts in series:
        if band_pts and len(band_pts) >= 2:
            top = " ".join(f"{X(x):.1f},{Y(hi):.1f}" for x, _, hi in band_pts)
            bot = " ".join(f"{X(x):.1f},{Y(lo):.1f}" for x, lo, _ in reversed(band_pts))
            s.append(f'<polygon class="{_cls(cfg["name"])} band" points="{top} {bot}"/>')

    hover = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "W": W, "H": H,
             "rates": [{"v": rv, "px": round(X(rv), 1)} for rv in all_rates], "series": []}
    end_labels = []
    for cfg, pts, band_pts in series:
        cls = _cls(cfg["name"])
        if len(pts) >= 2:
            d = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
            s.append(f'<path class="{cls} ln" d="{d}"/>')
        for x, y in pts:
            s.append(f'<circle class="{cls} mk" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.4"/>')
        if pts:
            lx, ly = pts[-1]
            end_labels.append([Y(ly), cls, cfg["name"], X(lx)])
        vmap = {round(float(x), 6): y for x, y in pts}
        hover["series"].append({"cls": cls, "name": cfg["name"],
            "vals": [None if round(rv, 6) not in vmap else round(vmap[round(rv, 6)], 2)
                     for rv in all_rates]})
    end_labels.sort(key=lambda e: e[0])
    for i in range(1, len(end_labels)):
        if end_labels[i][0] - end_labels[i - 1][0] < 12:
            end_labels[i][0] = end_labels[i - 1][0] + 12
    for ly_px, cls, name, x_px in end_labels:
        ly_px = min(ly_px, y1)
        s.append(f'<text class="{cls} endlab" x="{x_px+7:.1f}" y="{ly_px+3.5:.1f}">'
                 f'{html_lib.escape(name)}</text>')
    s.append("</svg>")
    return "".join(s), hover


def _nice_step(raw):
    import math
    if raw <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _tick(v):
    return f"{int(v)}" if v == int(v) else f"{v:g}"


# ---------------------------------------------------------------------------
# data-driven "what we learned" sentences (honest, matched-load comparisons)
# ---------------------------------------------------------------------------
def batching_finding(byname):
    s4, s32, s128 = byname.get("seqs-4"), byname.get("seqs-32"), byname.get("seqs-128")
    if not all(c and _rows_ok(c) for c in (s4, s32, s128)):
        return "max-num-seqs trades latency for throughput."
    s4_p99 = max((y for _, y in _xy(s4, "ttft_p99_ms")), default=0)
    s4_wait = max((r.get("waiting_max") or 0) for r in s4["rows"])
    peaks = {n: _peak(byname[n], "out_tok_s") for n in ("seqs-4", "seqs-32", "seqs-128")}
    best = max(peaks, key=lambda n: peaks[n] or 0)
    return (f"There is no single best setting. <b>max-num-seqs=4</b> starves under load — up to "
            f"{s4_wait:.0f} requests queue and the first-token wait blows past "
            f"<b>{s4_p99/1000:.1f} seconds</b>. <b>max-num-seqs=128</b> swings the other way: it "
            f"accepts so much work that the little GPU thrashes and throughput <b>collapses</b> "
            f"from {_peak(s128,'out_tok_s'):.0f} to {_at_max_rate(s128,'out_tok_s'):.0f} tok/s at the "
            f"heaviest load. <b>{best}</b> is the sweet spot, sustaining the most work "
            f"({peaks[best]:.0f} tok/s) without falling over.")


def memory_finding(byname):
    a, b = byname.get("seqs-128"), byname.get("mem-0.45")
    if not (a and b and _rows_ok(a) and _rows_ok(b)):
        return "A tighter KV budget only bites once the cache fills."
    pre = sum(r.get("preemptions") or 0 for r in b["rows"])
    kv_a, kv_b = _peak(a, "kv_cache_max_pct"), _peak(b, "kv_cache_max_pct")
    t_a, t_b = _peak(a, "out_tok_s"), _peak(b, "out_tok_s")
    if pre > 0:
        return (f"Halving the budget pushed peak cache use from {kv_a:.0f}% to {kv_b:.0f}% and "
                f"triggered {pre:.0f} preemptions.")
    return (f"Cutting the budget from 0.60 to 0.45 did push peak cache use up "
            f"({kv_a:.0f}% → {kv_b:.0f}% of budget), but it <b>never filled</b>: zero preemptions, "
            f"and throughput barely moved ({t_a:.0f} ≈ {t_b:.0f} tok/s). The lesson: on this GPU the "
            f"bottleneck is <b>compute</b>, not memory — the chip runs out of speed long before the "
            f"cache runs out of room. Memory tuning only pays off once you are actually memory-bound.")


def kv_finding(byname):
    a, b = byname.get("mem-0.45"), byname.get("fp8-0.45")
    if not (a and b and _rows_ok(a) and _rows_ok(b)):
        return "fp8 KV cache halves the KV footprint."
    rr = _fp8_kv_ratio(a, b)
    if rr is None:
        return "fp8 KV cache halves the KV footprint."
    ratio, rt, va, vb = rr
    t_a, t_b = _at_rate(a, "out_tok_s", rt), _at_rate(b, "out_tok_s", rt)
    cost = ""
    if t_a and t_b:
        cost = (f" with no throughput penalty ({t_a:.0f} vs {t_b:.0f} tok/s at that load — the gap "
                f"is within single-run noise)" if t_b >= t_a * 0.97
                else f" for a {(1-t_b/t_a)*100:.0f}% throughput cost")
    return (f"At the same 0.45 budget and the same load ({rt:.0f} req/s), fp8 held the same work in "
            f"<b>{ratio:.0f}% of the memory</b> ({va:.0f}% → {vb:.0f}% of budget){cost}. That freed "
            f"space is headroom you would spend on longer prompts or more concurrent users — it just "
            f"doesn't help <em>here</em>, because (see Experiment 2) memory was never what held us back.")


FINDING = {"batching": batching_finding, "memory": memory_finding, "kvquant": kv_finding}


# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------
def cmd_block(cfg, highlight):
    """Exact vLLM launch command for one config, with the distinguishing flag marked."""
    parts = [("VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen2.5-0.5B-Instruct-AWQ", False),
             ("  --max-model-len 4096", False),
             (f"  --gpu-memory-utilization {cfg['gpu_util']}", highlight == "--gpu-memory-utilization"),
             (f"  --max-num-seqs {cfg['max_num_seqs']}", highlight == "--max-num-seqs")]
    if cfg.get("kv_dtype", "auto") != "auto":
        parts.append((f"  --kv-cache-dtype {cfg['kv_dtype']}", highlight == "--kv-cache-dtype"))
    elif highlight == "--kv-cache-dtype":
        parts.append(("  # (no --kv-cache-dtype flag: defaults to 16-bit)", True))
    lines = []
    for txt, hl in parts:
        e = html_lib.escape(txt)
        lines.append(f'<span class="hl">{e}</span>' if hl else e)
    head = (f'<div class="cmdhead"><span class="sw {_cls(cfg["name"])}"></span>'
            f'<b>{html_lib.escape(cfg["name"])}</b> — {html_lib.escape(cfg["label"])}</div>')
    body = " \\\n".join(lines)
    return f'<div class="cmd">{head}<pre>{body}</pre></div>'


def _tile(k, v, unit, note, cls):
    return (f'<div class="tile"><div class="tk">{html_lib.escape(k)}</div>'
            f'<div class="tv {cls}">{v}<span class="tu">{html_lib.escape(unit)}</span></div>'
            f'<div class="tn">{note}</div></div>')


def _panels_grid(sec, cfgs, all_rates, hovers):
    out = []
    for pk in sec["panels"]:
        spec = PANELS[pk]
        cid = f'{sec["id"]}_{pk}'
        svg, hv = line_panel(spec, cfgs, all_rates)
        hovers[cid] = hv
        out.append(
            f'<figure class="panel" data-chart="{cid}">'
            f'<figcaption class="ptitle">{html_lib.escape(spec["title"])} '
            f'<span class="punit">({html_lib.escape(spec["unit"])})</span></figcaption>'
            f'<p class="pexplain">{html_lib.escape(spec["explain"])}</p>'
            f'<div class="svgbox">{svg}<div class="tt" id="tt-{cid}"></div></div></figure>')
    return f'<div class="panels">{"".join(out)}</div>'


# ---------------------------------------------------------------------------
def write_comparison_report(summary: dict, path: Path):
    meta = summary["meta"]
    byname = _by_name(summary)
    all_rates = sorted({float(x) for x in str(meta["rates"]).split(",")})
    hovers = {}

    # per-config theme-aware colour classes
    color_css = []
    for c in summary["configs"]:
        sel = "." + _cls(c["name"])
        color_css.append(f"{sel}{{color:{c['color_light']};}}")
        color_css.append(f"@media (prefers-color-scheme:dark){{{sel}{{color:{c['color_dark']};}}}}")
    color_css = "\n".join(color_css)

    # ---- headline tiles ----
    ok_cfgs = [c for c in summary["configs"] if _rows_ok(c)]
    tiles = []
    if ok_cfgs:
        best = max(ok_cfgs, key=lambda c: _peak(c, "out_tok_s") or 0)
        tiles.append(_tile("Most tokens/sec we saw", f"{_peak(best,'out_tok_s'):.0f}", "tok/s",
                           f'from <b class="{_cls(best["name"])}">{html_lib.escape(best["name"])}</b>, '
                           f'the batch that stayed stable', "muted-tile"))
    a, b = byname.get("mem-0.45"), byname.get("fp8-0.45")
    if a and b and _rows_ok(a) and _rows_ok(b):
        rr = _fp8_kv_ratio(a, b)
        if rr:
            ratio, rt, va, vb = rr
            tiles.append(_tile("fp8 cache footprint", f"{ratio:.0f}", "% of 16-bit",
                               f"same work, {va:.0f}%→{vb:.0f}% of the budget", _cls("fp8-0.45")))
    total_preempt = sum((r.get("preemptions") or 0) for c in ok_cfgs for r in c["rows"])
    kv_all = [k for k in (_peak(c, "kv_cache_max_pct") for c in ok_cfgs) if k is not None]
    if kv_all:
        tiles.append(_tile("Memory evictions", "0" if total_preempt == 0 else f"{total_preempt:.0f}",
                           "preemptions",
                           f"peak cache use only {max(kv_all):.0f}% — never memory-bound"
                           if total_preempt == 0 else "cache filled under the tight budget",
                           "muted-tile"))
    tiles_html = f'<div class="tiles">{"".join(tiles)}</div>' if tiles else ""

    # ---- experiment sections ----
    sections_html = []
    for sec in SECTIONS:
        cfgs = [byname[n] for n in sec["configs"] if n in byname and _rows_ok(byname[n])]
        if not cfgs:
            continue
        cmds = "".join(cmd_block(byname[n], sec["highlight"]) for n in sec["configs"] if n in byname)
        legend = "".join(
            f'<span class="lg"><span class="sw {_cls(c["name"])}"></span>'
            f'<b>{html_lib.escape(c["name"])}</b> <span class="lgsub">{html_lib.escape(c["label"])}</span></span>'
            for c in cfgs)
        sections_html.append(f"""
<section class="lesson" id="{sec['id']}">
  <div class="lnum">{sec['num']}</div>
  <h2>{html_lib.escape(sec['title'])}</h2>
  <div class="knob">the knob: <code>{html_lib.escape(sec['knob'])}</code></div>
  <p class="prose">{sec['what']}</p>
  <p class="prose">{sec['why']}</p>
  <h3 class="mini">The exact commands we compared</h3>
  <p class="prose sm">Everything else is identical — only the highlighted flag changes between runs.</p>
  <div class="cmds">{cmds}</div>
  <h3 class="mini">What the run showed</h3>
  <div class="legendrow">{legend}</div>
  {_panels_grid(sec, cfgs, all_rates, hovers)}
  <div class="takeaway"><span class="tw-lab">What we learned</span><p>{FINDING[sec['id']](byname)}</p></div>
</section>""")

    # ---- data table ----
    tcols = [("config", ""), ("rate", "req/s"), ("req_s", "req/s"), ("out_tok_s", "tok/s"),
             ("ttft_p50_ms", "ms"), ("ttft_p99_ms", "ms"), ("itl_mean_ms", "ms/t"),
             ("kv_cache_max_pct", "%"), ("waiting_max", "queued"), ("preemptions", "evict")]
    thead = "".join(f"<th>{html_lib.escape(c)}<span class='u'>{html_lib.escape(u)}</span></th>"
                    for c, u in tcols)
    trows = []
    for c in summary["configs"]:
        for i, r in enumerate(c.get("rows", [])):
            cells = []
            for col, _ in tcols:
                if col == "config":
                    tag = (f'<span class="sw {_cls(c["name"])}"></span>{html_lib.escape(c["name"])}'
                           if i == 0 else "")
                    cells.append(f'<td class="cfgcell">{tag}</td>')
                else:
                    v = r.get(col)
                    nd = 2 if col in ("rate", "req_s", "itl_mean_ms") else \
                        (1 if col in ("ttft_p50_ms", "ttft_p99_ms", "kv_cache_max_pct") else 0)
                    cells.append(f"<td>{_fmt(v, nd) if v is not None else '—'}</td>")
            trows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (f'<div class="table-wrap"><table><thead><tr>{thead}</tr></thead>'
                  f'<tbody>{"".join(trows)}</tbody></table></div>')

    # ---- full command list (method) ----
    all_cmds = "".join(cmd_block(c, None) for c in summary["configs"])

    meta_items = "".join(
        f'<div class="meta-item"><span class="meta-k">{html_lib.escape(k)}</span>'
        f'<span class="meta-v">{html_lib.escape(str(v))}</span></div>'
        for k, v in [("model", meta["model"]), ("GPU", meta["gpu"]),
                     ("vLLM", meta["vllm_version"]),
                     ("each request", f'{meta["prompt_tokens"]}-token prompt → {meta["output_tokens"]} tokens out'),
                     ("load levels tested", f'{meta["rates"]} requests/sec'),
                     ("measured per level", f'{meta["max_seconds"]} s'),
                     ("generated", meta["generated"])])

    body = _CONCEPTS + f"""
<section class="rig">
  <h2>The test rig</h2>
  <p class="prose">Every number on this page comes from one small language model
  (<b>Qwen2.5-0.5B</b>, a 0.5-billion-parameter model, AWQ-quantized) served by <b>vLLM</b> on a
  <b>4&nbsp;GB laptop GPU</b> (an RTX&nbsp;3050 under WSL2). A tiny model on a tiny card is deliberate:
  the effects below are the same ones you meet on an 80&nbsp;GB datacentre GPU, just at a scale you can
  run at home. To drive traffic we use <b>GuideLLM</b>, which fires synthetic requests at a fixed rate
  and measures what comes back.</p>
  <p class="prose">The single most important idea for reading the charts: <b>we sweep the offered load</b>
  — the number of requests arriving per second, along the bottom of every chart — from gentle (1/s) to
  overloaded (12/s). A server can look perfect until traffic rises, so we watch how each setting behaves
  <em>as the pressure climbs</em>. Every configuration is driven with the exact same requests, so the
  lines are directly comparable.</p>
  <div class="meta">{meta_items}</div>
  {tiles_html}
</section>
{''.join(sections_html)}
<section class="lesson" id="tuning">
  <div class="lnum">★</div>
  <h2>Putting it together — how to actually tune a server</h2>
  <p class="prose">The three experiments give a simple recipe, in order:</p>
  <ol class="steps">
    <li><b>Start with the batch cap (<code>--max-num-seqs</code>).</b> It is the biggest lever on both
      latency and throughput. Too small and requests queue until the first-token wait becomes seconds;
      too large and a small GPU thrashes and throughput collapses. Find the largest cap that still holds
      your latency target under peak load.</li>
    <li><b>Decide what "good" means with a latency budget (an SLO).</b> Pick a ceiling for the p99
      first-token wait — say 500&nbsp;ms — and read across each throughput chart to the busiest point
      still under that ceiling. That, not the raw peak, is your usable throughput.</li>
    <li><b>Only tune memory once you are memory-bound.</b> If you never see preemptions and cache use
      stays well under 100% (as here), more memory won't help — you are compute-bound, so buy compute or
      pick a smaller model. When you <em>do</em> see preemptions, raise <code>--gpu-memory-utilization</code> or
      turn on <code>--kv-cache-dtype fp8</code> to roughly double the cache for free.</li>
  </ol>
  <p class="prose">Which knob dominates depends on <em>your</em> model, GPU and traffic — the value of a page
  like this is that it shows you how to find out, not a universal answer.</p>
</section>
<section class="lesson" id="data">
  <h2>Every measurement</h2>
  <p class="prose sm">One row per configuration per load level. <b>queued</b> is the most requests waiting
  at once (a sign the server can't keep up); <b>evict</b> is preemptions. This is the raw data behind
  every chart above.</p>
  {table_html}
</section>
<section class="lesson" id="method">
  <h2>Exact commands & honest caveats</h2>
  <p class="prose sm">Base for all runs — only the flags noted in each experiment differ. The
  <code>VLLM_USE_V2_MODEL_RUNNER=0</code> prefix is a WSL2 workaround (the newer runner needs a GPU
  feature WSL doesn't expose).</p>
  <div class="cmds">{all_cmds}</div>
  <p class="prose sm"><b>Caveats.</b> Each configuration was measured <b>once</b> (n&nbsp;=&nbsp;1), so treat
  small differences — a few percent of throughput — as noise; the large effects (a batch starving, a batch
  collapsing, fp8 halving the cache) are real. Results are specific to this model, this 4&nbsp;GB GPU and
  this short-prompt workload. KV-cache, queue depth and preemptions are read from vLLM's own
  <code>/metrics</code> endpoint while the load runs.</p>
</section>"""

    doc = _SHELL.format(color_css=color_css, body=body, hover_json=json.dumps(hovers))
    Path(path).write_text(doc)


# ---------------------------------------------------------------------------
# static teaching content: the primer + a prefill/decode timeline diagram
# ---------------------------------------------------------------------------
_TIMELINE = """
<svg viewBox="0 0 640 150" class="timeline" role="img"
     aria-label="Timeline of one request: queue, prefill to first token, then decoding one token at a time">
  <line x1="20" y1="96" x2="620" y2="96" class="tl-axis"/>
  <rect x="20"  y="74" width="70"  height="22" rx="3" class="tl-queue"/>
  <text x="55" y="89" class="tl-in">queue</text>
  <rect x="90"  y="74" width="120" height="22" rx="3" class="tl-prefill"/>
  <text x="150" y="89" class="tl-in">PREFILL</text>
  <circle cx="210" cy="96" r="7" class="tl-tok tl-first"/>
  <text x="210" y="122" class="tl-tl">token 1</text>
  <g class="tl-dec">
    <rect x="210" y="80" width="70" height="16" rx="3" class="tl-decode"/>
    <circle cx="280" cy="96" r="6" class="tl-tok"/>
    <rect x="280" y="80" width="70" height="16" rx="3" class="tl-decode"/>
    <circle cx="350" cy="96" r="6" class="tl-tok"/>
    <rect x="350" y="80" width="70" height="16" rx="3" class="tl-decode"/>
    <circle cx="420" cy="96" r="6" class="tl-tok"/>
    <text x="330" y="122" class="tl-tl">token 2, 3, 4 …  (streamed)</text>
  </g>
  <line x1="20" y1="46" x2="210" y2="46" class="tl-brk"/>
  <line x1="20" y1="42" x2="20" y2="50" class="tl-brk"/>
  <line x1="210" y1="42" x2="210" y2="50" class="tl-brk"/>
  <text x="115" y="38" class="tl-lab tl-ttft">TTFT — time to first token (the lag you feel)</text>
  <line x1="280" y1="60" x2="350" y2="60" class="tl-brk2"/>
  <line x1="280" y1="56" x2="280" y2="64" class="tl-brk2"/>
  <line x1="350" y1="56" x2="350" y2="64" class="tl-brk2"/>
  <text x="315" y="52" class="tl-lab tl-itl">ITL — gap between tokens</text>
  <text x="620" y="140" class="tl-time" text-anchor="end">time  →</text>
</svg>"""

_CONCEPTS = f"""
<section class="intro">
  <h1>Understanding LLM serving — latency, throughput &amp; memory</h1>
  <p class="lede">Serving a large language model is a balancing act between three things: how fast each
  user gets an answer (<b>latency</b>), how many users you can serve at once (<b>throughput</b>), and how
  much <b>GPU memory</b> the whole thing needs. This page explains what those mean from scratch, then runs
  three real experiments on <code>vLLM</code> to show how a handful of server settings move them around.</p>
</section>
<section class="primer">
  <h2>First: what happens when a model answers</h2>
  <p class="prose">When you send a prompt, the server does the work in <b>two distinct phases</b>:</p>
  <div class="phase-grid">
    <div class="phase"><div class="ph-k">Phase 1 · Prefill</div>
      <p>The model reads your whole prompt in a single pass and produces the <b>first</b> token. This is
      compute-heavy but parallel, and it is what you wait through before <em>anything</em> appears.</p></div>
    <div class="phase"><div class="ph-k">Phase 2 · Decode</div>
      <p>The model then generates the rest <b>one token at a time</b>, each step reading everything written
      so far. These tokens stream out; the gap between them is what makes text feel fast or sluggish.</p></div>
  </div>
  {_TIMELINE}
  <p class="prose">That timeline gives us the two latency numbers used throughout this page, plus the two
  throughput and memory numbers that describe the server as a whole:</p>
  <dl class="glossary">
    <dt>TTFT — time to first token</dt>
    <dd>The prefill wait (plus any time spent queued). This is the lag a user feels before the answer starts.
      We track its typical value (<b>p50</b>) and its near-worst-case tail (<b>p99</b>) — the tail is what
      makes a service feel unreliable even when the average looks fine.</dd>
    <dt>ITL — inter-token latency</dt>
    <dd>The decode gap between streamed tokens. Low ITL is a smooth, fast-flowing response.</dd>
    <dt>Throughput — tokens/sec &amp; requests/sec</dt>
    <dd>How much work the whole server gets done: total tokens emitted per second across every user at once.
      This is what you are paying the GPU for, and it usually <em>rises</em> with more simultaneous users —
      up to a point.</dd>
    <dt>Offered load — requests/sec</dt>
    <dd>How much traffic is arriving. Every chart sweeps this along the bottom, from light to overloaded, so
      you can see where a setting stops coping.</dd>
    <dt>The KV-cache — the memory that inference actually spends</dt>
    <dd>During decode, the model would have to re-read the entire sequence for every new token. Instead it
      caches each token's attention state (its "keys" and "values") in GPU memory — the <b>KV-cache</b> — so
      each new token only looks at the cache. That cache grows with every token of every in-flight request,
      so <b>more users and longer contexts cost more memory</b>. It lives in the GPU alongside the model
      weights, and when it fills, the server must <b>preempt</b> — evict a running request and redo it later —
      which spikes latency. Fitting more into it is the whole point of the memory experiments below.</dd>
  </dl>
  <div class="keyidea"><span class="ki-lab">The core tension</span>
    <p>Running more requests together fills the GPU and lifts throughput — but makes every user's tokens wait
    for the batch, and eats KV-cache memory. Push too hard and requests queue (latency explodes) or the cache
    overflows (preemptions). Tuning a server is finding the point just before that edge. The three experiments
    below each turn one knob and watch where the edge is.</p></div>
</section>"""


# ---------------------------------------------------------------------------
_SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Understanding LLM serving — a vLLM walkthrough</title>
<style>
:root {{
  color-scheme: light dark;
  --plane:#f6f6f3; --surface:#fcfcfb; --raise:#f0efeb; --ink:#111110; --ink2:#4b4a47;
  --muted:#8a8880; --grid:#e6e5de; --axis:#c3c2b7; --border:rgba(17,17,16,0.11);
  --accent:#2a78d6; --band:0.13; --code:#2b2a28; --codebg:#f3f2ee;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --plane:#0c0c0c; --surface:#191918; --raise:#232321; --ink:#f4f4f2; --ink2:#c3c2b7;
    --muted:#8a8880; --grid:#2c2c2a; --axis:#3a3a37; --border:rgba(255,255,255,0.11);
    --accent:#3987e5; --band:0.20; --code:#d8d7d2; --codebg:#111110;
  }}
}}
{color_css}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.6;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:880px; margin:0 auto; padding:44px 24px 80px; }}
h1 {{ font-size:30px; font-weight:680; line-height:1.2; margin:0 0 14px; letter-spacing:-0.01em; }}
h2 {{ font-size:22px; font-weight:660; margin:0 0 10px; line-height:1.25; letter-spacing:-0.01em; }}
h3.mini {{ font-size:14px; font-weight:660; margin:26px 0 6px; text-transform:uppercase;
  letter-spacing:.05em; color:var(--ink2); }}
.lede {{ font-size:17px; color:var(--ink2); margin:0; }}
.prose {{ font-size:15.5px; color:var(--ink2); margin:0 0 14px; }}
.prose.sm {{ font-size:13.5px; }}
.prose b, .glossary b, .keyidea b, .steps b {{ color:var(--ink); font-weight:640; }}
code {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9em; color:var(--code);
  background:var(--codebg); padding:1px 5px; border-radius:5px; border:1px solid var(--border); }}
section {{ margin-bottom:34px; }}
.intro {{ margin-bottom:26px; }}
.primer {{ border-top:1px solid var(--border); padding-top:26px; }}
/* two-phase cards */
.phase-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:14px 0 20px; }}
.phase {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px 17px; }}
.phase p {{ font-size:14px; color:var(--ink2); margin:6px 0 0; }}
.ph-k {{ font-size:12.5px; font-weight:680; color:var(--accent); text-transform:uppercase; letter-spacing:.03em; }}
/* timeline diagram */
.timeline {{ width:100%; height:auto; display:block; margin:6px 0 20px; overflow:visible; }}
.tl-axis {{ stroke:var(--axis); stroke-width:1.5; }}
.tl-queue {{ fill:var(--raise); stroke:var(--border); }}
.tl-prefill {{ fill:var(--accent); opacity:.85; }}
.tl-decode {{ fill:var(--accent); opacity:.30; }}
.tl-tok {{ fill:var(--accent); stroke:var(--surface); stroke-width:1.5; }}
.tl-first {{ fill:var(--accent); }}
.tl-in {{ fill:#fff; font-size:11px; font-weight:640; text-anchor:middle; }}
.tl-queue + .tl-in {{ fill:var(--ink2); }}
.tl-tl {{ fill:var(--muted); font-size:10.5px; text-anchor:middle; }}
.tl-brk {{ stroke:var(--accent); stroke-width:1.3; }}
.tl-brk2 {{ stroke:var(--muted); stroke-width:1.1; }}
.tl-lab {{ font-size:11px; font-weight:620; text-anchor:middle; }}
.tl-ttft {{ fill:var(--accent); }} .tl-itl {{ fill:var(--muted); }}
.tl-time {{ fill:var(--muted); font-size:10.5px; }}
/* glossary */
.glossary {{ margin:8px 0 20px; }}
.glossary dt {{ font-size:15px; font-weight:660; color:var(--ink); margin-top:14px; }}
.glossary dd {{ font-size:14.5px; color:var(--ink2); margin:3px 0 0; padding-left:14px;
  border-left:2px solid var(--border); }}
.keyidea {{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--accent);
  border-radius:10px; padding:14px 18px; margin-top:8px; }}
.keyidea p {{ font-size:14.5px; color:var(--ink2); margin:6px 0 0; }}
.ki-lab, .tw-lab {{ font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); }}
/* rig + meta + tiles */
.rig {{ border-top:1px solid var(--border); padding-top:26px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 26px; padding:14px 18px; margin:18px 0 14px;
  background:var(--surface); border:1px solid var(--border); border-radius:10px; }}
.meta-item {{ display:flex; flex-direction:column; }}
.meta-k {{ font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
.meta-v {{ font-size:13px; color:var(--ink); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px 17px; }}
.tk {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
.tv {{ font-size:27px; font-weight:680; font-variant-numeric:tabular-nums; margin-top:3px; }}
.tv.muted-tile {{ color:var(--ink); }}
.tv .tu {{ font-size:12px; color:var(--ink2); margin-left:4px; font-weight:400; }}
.tn {{ font-size:12px; margin-top:3px; color:var(--ink2); }}
/* lessons */
.lesson {{ position:relative; background:var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:24px 26px; margin-bottom:20px; }}
.lnum {{ position:absolute; top:-14px; left:22px; width:30px; height:30px; border-radius:50%;
  background:var(--accent); color:#fff; font-weight:700; font-size:15px; display:flex;
  align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,.2); }}
.lesson h2 {{ margin-top:6px; }}
.knob {{ display:inline-block; font-size:12px; color:var(--muted); margin:0 0 14px; }}
.knob code {{ font-size:12.5px; }}
/* command blocks */
.cmds {{ display:flex; flex-direction:column; gap:10px; margin:8px 0 6px; }}
.cmd {{ background:var(--codebg); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.cmdhead {{ font-size:12.5px; padding:8px 12px; border-bottom:1px solid var(--border); color:var(--ink); }}
.cmd pre {{ margin:0; padding:11px 13px; font-family:ui-monospace,Menlo,Consolas,monospace;
  font-size:12px; line-height:1.55; color:var(--code); overflow-x:auto; white-space:pre; }}
.cmd .hl {{ background:rgba(42,120,214,.16); color:var(--ink); border-radius:4px;
  padding:0 3px; font-weight:640; box-decoration-break:clone; -webkit-box-decoration-break:clone; }}
/* legend + panels */
.legendrow {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:4px 0 14px; }}
.lg {{ font-size:12.5px; color:var(--ink2); }}
.lg b {{ color:var(--ink); font-weight:640; }} .lgsub {{ color:var(--muted); }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px;
  vertical-align:-1px; background:currentColor; }}
.panels {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px 22px; }}
.panel {{ margin:0; min-width:0; }}
.ptitle {{ font-size:13px; font-weight:660; color:var(--ink); }}
.punit {{ color:var(--muted); font-weight:400; }}
.pexplain {{ font-size:12px; color:var(--muted); margin:2px 0 6px; line-height:1.45; }}
.svgbox {{ position:relative; }}
svg.lc {{ width:100%; height:auto; display:block; overflow:visible; }}
.cgrid {{ stroke:var(--grid); stroke-width:1; }}
.caxis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:10px; font-variant-numeric:tabular-nums; }}
.alab {{ fill:var(--ink2); font-size:10.5px; font-weight:500; }}
.ln {{ fill:none; stroke:currentColor; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.mk {{ fill:currentColor; stroke:var(--surface); stroke-width:1.4; }}
.band {{ fill:currentColor; stroke:none; opacity:var(--band); }}
.endlab {{ fill:currentColor; font-size:10.5px; font-weight:700; }}
.tt {{ position:absolute; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:7px 9px;
  font-size:11.5px; box-shadow:0 4px 14px rgba(0,0,0,.16); z-index:5; min-width:120px; }}
.tt .tth {{ font-weight:660; margin-bottom:4px; font-variant-numeric:tabular-nums; }}
.tt .ttr {{ display:flex; justify-content:space-between; gap:14px; align-items:center; }}
.tt .ttr b {{ font-variant-numeric:tabular-nums; }}
.crosshair {{ stroke:var(--axis); stroke-width:1; stroke-dasharray:3 3; pointer-events:none; }}
/* takeaway */
.takeaway {{ background:var(--raise); border:1px solid var(--border); border-radius:10px;
  padding:13px 16px; margin-top:18px; }}
.takeaway p {{ font-size:14.5px; color:var(--ink); margin:5px 0 0; }}
/* steps */
.steps {{ font-size:15px; color:var(--ink2); padding-left:22px; margin:8px 0 14px; }}
.steps li {{ margin-bottom:10px; }}
/* table */
.table-wrap {{ overflow-x:auto; margin-top:6px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }}
th, td {{ padding:6px 9px; text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums;
  border-bottom:1px solid var(--grid); }}
th {{ color:var(--muted); font-weight:600; text-transform:uppercase; font-size:9.5px; letter-spacing:.03em; }}
th .u {{ display:block; font-weight:400; text-transform:none; letter-spacing:0; color:var(--axis); }}
th:first-child, td:first-child {{ text-align:left; }}
.cfgcell {{ font-weight:640; }}
.foot {{ color:var(--muted); font-size:12px; }}
</style></head>
<body><div class="wrap">
{body}
</div>
<script id="hoverdata" type="application/json">{hover_json}</script>
<script>
const HOV = JSON.parse(document.getElementById('hoverdata').textContent);
document.querySelectorAll('.panel').forEach(fig => {{
  const id = fig.dataset.chart, d = HOV[id]; if(!d) return;
  const svg = fig.querySelector('svg'), tt = fig.querySelector('.tt');
  const ch = document.createElementNS('http://www.w3.org/2000/svg','line');
  ch.setAttribute('class','crosshair'); ch.setAttribute('y1',d.y0); ch.setAttribute('y2',d.y1);
  ch.style.opacity=0; svg.appendChild(ch);
  const overlay = document.createElementNS('http://www.w3.org/2000/svg','rect');
  overlay.setAttribute('x',d.x0); overlay.setAttribute('y',d.y0);
  overlay.setAttribute('width',d.x1-d.x0); overlay.setAttribute('height',d.y1-d.y0);
  overlay.setAttribute('fill','transparent'); svg.appendChild(overlay);
  function move(ev){{
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * d.W;
    let bi=0, bd=1e9;
    d.rates.forEach((rt,i)=>{{const dist=Math.abs(rt.px-px); if(dist<bd){{bd=dist;bi=i;}}}});
    const rt = d.rates[bi];
    ch.setAttribute('x1',rt.px); ch.setAttribute('x2',rt.px); ch.style.opacity=1;
    let rows='';
    d.series.forEach(s=>{{const v=s.vals[bi];
      rows += `<div class="ttr"><span><span class="sw ${{s.cls}}"></span>${{s.name}}</span>`+
              `<b>${{v==null?'—':v.toLocaleString()}}</b></div>`;}});
    tt.innerHTML = `<div class="tth">${{rt.v}} req/s</div>${{rows}}`;
    tt.style.opacity=1;
    const bx = rt.px / d.W * r.width;
    tt.style.left = Math.min(Math.max(bx-tt.offsetWidth/2,2), r.width-tt.offsetWidth-2)+'px';
    tt.style.top = '4px';
  }}
  overlay.addEventListener('pointermove',move);
  overlay.addEventListener('pointerleave',()=>{{tt.style.opacity=0; ch.style.opacity=0;}});
}});
</script>
</body></html>"""


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_results/configs_summary.json")
    data = json.loads(src.read_text())
    out = src.parent / "configs_comparison.html"
    write_comparison_report(data, out)
    print(f"wrote {out}")
