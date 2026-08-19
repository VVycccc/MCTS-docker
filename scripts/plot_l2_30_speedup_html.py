#!/usr/bin/env python3
"""Generate the L2 30-problem speedup chart (HTML), mirroring l1_speedup_chart.html
but with the geomean stat tiles removed (no averages) and the fixed-run data.

Reads: output/full_mcts_l2_search_fixed_20260810/l2_30_results.json
Writes: figures/l2_30_speedup_chart.html
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "output/full_mcts_l2_search_fixed_20260810/l2_30_results.json"
OUTPUT = ROOT / "figures/l2_30_speedup_chart.html"


def main() -> None:
    data = json.loads(INPUT.read_text())
    rows = data["rows"]

    # sort: vs-PyTorch desc, None last
    def keypy(r):
        v = r.get("speedup_vs_pytorch")
        return (v is None, -(v if isinstance(v, (int, float)) and math.isfinite(v) else 0.0))

    rows = sorted(rows, key=keypy)

    valid = [r for r in rows if isinstance(r.get("speedup_vs_pytorch"), (int, float))]
    beats = sum(1 for r in valid if r["speedup_vs_pytorch"] > 1.0)
    best_py = max((r["speedup_vs_pytorch"] for r in valid), default=None)
    best_sd = max(
        (r["speedup_vs_seed"] for r in rows if isinstance(r.get("speedup_vs_seed"), (int, float))),
        default=None,
    )

    def fmt(v):
        if v is None:
            return "—"
        if v >= 100:
            return f"{v:.0f}"
        if v >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    # emit JS rows array
    js_rows = []
    for r in rows:
        def n(v):
            return None if v is None else round(float(v), 4)
        js_rows.append(
            "    {name:%s, vsPy:%s, vsSeed:%s, baseMs:%s, seedMs:%s, champMs:%s, fam:%s}"
            % (
                json.dumps(r["problem"]),
                "null" if r.get("speedup_vs_pytorch") is None else n(r["speedup_vs_pytorch"]),
                "null" if r.get("speedup_vs_seed") is None else n(r["speedup_vs_seed"]),
                "null" if r.get("pytorch_latency_ms") is None else round(float(r["pytorch_latency_ms"]), 4),
                "null" if r.get("seed_latency_ms") is None else round(float(r["seed_latency_ms"]), 4),
                "null" if r.get("champion_latency_ms") is None else round(float(r["champion_latency_ms"]), 4),
                json.dumps(r.get("family", "")),
            )
        )
    js_rows_text = ",\n".join(js_rows)

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DirecTune-MCTS L2 (30) — Speedup vs PyTorch &amp; vs Seed</title>
<style>
  :root {{ color-scheme: light dark; }}
  .viz-root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --surface-2:      #f3f3f0;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #8a897f;
    --rule:           #d8d7d0;
    --series-py:      #2a78d6;
    --series-seed:    #eb6834;
    --ref-line:       #b7b6ad;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --surface-2:      #232322;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #8f8e83;
      --rule:           #3a3a37;
      --series-py:      #3987e5;
      --series-seed:    #d95926;
      --ref-line:       #54534a;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --surface-2:      #232322;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #8f8e83;
    --rule:           #3a3a37;
    --series-py:      #3987e5;
    --series-seed:    #d95926;
    --ref-line:       #54534a;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--surface-2); }}
  body {{
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, system-ui, sans-serif;
    color: var(--text-primary);
    padding: 28px 20px 48px;
  }}
  .viz-root {{ max-width: 1280px; margin: 0 auto; background: var(--surface-1); border-radius: 10px; padding: 26px 28px 30px; }}
  h1 {{ font-size: 19px; margin: 0 0 4px; font-weight: 650; letter-spacing: -0.01em; }}
  .sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 20px; }}
  .stats {{ display: flex; gap: 26px; flex-wrap: wrap; margin: 0 0 22px; padding: 14px 16px; background: var(--surface-2); border-radius: 8px; }}
  .stat {{ display: flex; flex-direction: column; gap: 2px; }}
  .stat .k {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat .v {{ font-size: 18px; font-weight: 650; }}
  .stat .v .x {{ color: var(--text-muted); font-weight: 400; font-size: 14px; }}
  .legend {{ display: flex; gap: 18px; align-items: center; margin: 0 0 14px; font-size: 13px; color: var(--text-secondary); }}
  .legend .sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
  .legend .py {{ background: var(--series-py); }}
  .legend .sd {{ background: var(--series-seed); }}
  .legend .ref {{ display: inline-block; width: 16px; height: 0; border-top: 1.5px dashed var(--ref-line); margin-right: 6px; vertical-align: 3px; }}
  .chart-wrap {{ position: relative; overflow-x: auto; }}
  svg {{ display: block; }}
  .axis text {{ fill: var(--text-secondary); font-size: 11px; }}
  .axis line, .axis path {{ stroke: var(--rule); }}
  .grid line {{ stroke: var(--rule); stroke-dasharray: 2 3; opacity: 0.55; }}
  .grid path {{ stroke: none; }}
  .refline {{ stroke: var(--ref-line); stroke-width: 1.25; stroke-dasharray: 4 3; }}
  .bar-py {{ fill: var(--series-py); }}
  .bar-sd {{ fill: var(--series-seed); }}
  .ylabel {{ fill: var(--text-secondary); font-size: 12px; }}
  .row-label {{ fill: var(--text-primary); font-size: 10.5px; }}
  .row-label.muted {{ fill: var(--text-muted); }}
  .missing {{ fill: var(--text-muted); font-size: 10.5px; }}
  .tooltip {{
    position: absolute; pointer-events: none; opacity: 0; transition: opacity .12s;
    background: var(--surface-2); border: 1px solid var(--rule); border-radius: 6px;
    padding: 7px 10px; font-size: 12px; color: var(--text-primary); white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .tooltip .t {{ font-weight: 650; margin-bottom: 3px; }}
  .tooltip .r {{ color: var(--text-secondary); }}
  .tooltip .r .py {{ color: var(--series-py); font-weight: 600; }}
  .tooltip .r .sd {{ color: var(--series-seed); font-weight: 600; }}
  .foot {{ margin-top: 18px; font-size: 12px; color: var(--text-muted); line-height: 1.5; }}
  .foot code {{ background: var(--surface-2); padding: 1px 5px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="viz-root">
  <h1>DirecTune-MCTS · L2 kernel speedups (30 problems, fixed run)</h1>
  <div class="sub">30 L2 fusion problems · Triton champion (MCTS) vs PyTorch baseline and vs naive Triton seed · sorted by vs-PyTorch speedup · no averages shown</div>
  <div class="stats">
    <div class="stat"><span class="k">problems</span><span class="v">30</span></div>
    <div class="stat"><span class="k">valid triton</span><span class="v">29<span class="x">/30</span></span></div>
    <div class="stat"><span class="k">beats PyTorch</span><span class="v">{beats}<span class="x">/30</span></span></div>
    <div class="stat"><span class="k">best vs PyTorch</span><span class="v">{fmt(best_py)}<span class="x">×</span></span></div>
    <div class="stat"><span class="k">best vs seed</span><span class="v">{fmt(best_sd)}<span class="x">×</span></span></div>
  </div>
  <div class="legend">
    <span><span class="sw py"></span>vs PyTorch (baseline / champion)</span>
    <span><span class="sw sd"></span>vs naive seed (seed / champion)</span>
    <span><span class="ref"></span>1.0× parity line</span>
  </div>
  <div class="chart-wrap" id="cw">
    <svg id="chart" xmlns="http://www.w3.org/2000/svg"></svg>
    <div class="tooltip" id="tt"></div>
  </div>
  <div class="foot">
    Bars are log-scaled speedup ratios; longer bar = faster champion. Bars right of the dashed 1.0× line beat PyTorch; left of it the champion is slower than eager. <code>8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum</code> timed out (rc=124) → no final champion. Source: <code>output/full_mcts_l2_search_fixed_20260810/l2_30_results.json</code>.
  </div>
</div>
<script>
window.__ROWS = [
{js_rows_text}
];
(function(){{
  const rows = window.__ROWS;
  const svg = document.getElementById('chart');
  const tt = document.getElementById('tt');
  const cw = document.getElementById('cw');

  const W = 1240;
  const rowH = 22;
  const topPad = 18, botPad = 40, leftPad = 410, rightPad = 60;
  const H = topPad + rows.length * rowH + botPad;

  // log scale; ratio=1 at the parity line. domain [0.01, 1000] covers 0.015x..709x.
  const lo = Math.log(0.01), hi = Math.log(1000);
  const plotW = W - leftPad - rightPad;
  function xOf(ratio){{
    const t = (Math.log(ratio) - lo) / (hi - lo);
    return leftPad + t * plotW;
  }}
  const x1 = xOf(1);

  const ticks = [0.01, 0.1, 1, 10, 100, 1000];

  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);

  const ns = 'http://www.w3.org/2000/svg';
  function el(tag, attrs){{
    const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }}

  const grid = el('g', {{class:'grid'}});
  ticks.forEach(t => {{
    const x = xOf(t);
    grid.appendChild(el('line', {{x1:x, y1:topPad, x2:x, y2:H-botPad}}));
  }});
  svg.appendChild(grid);

  svg.appendChild(el('line', {{class:'refline', x1:x1, y1:topPad-4, x2:x1, y2:H-botPad+4}}));

  rows.forEach((r, i) => {{
    const y = topPad + i * rowH;
    const barH = 8;
    const gap = 2;
    const muted = (r.vsPy == null) || (r.vsPy < 1);
    // vs PyTorch (top bar)
    if (r.vsPy != null){{
      const yPy = y + 2;
      const xPy = xOf(r.vsPy);
      const wPy = Math.max(2, Math.abs(xPy - x1));
      const bPy = el('rect', {{class:'bar-py', x: Math.min(x1,xPy), y: yPy, width: wPy, height: barH, rx: 2}});
      bPy.dataset.idx = i; bPy.dataset.which = 'py';
      svg.appendChild(bPy);
    }}
    // vs seed (bottom bar)
    if (r.vsSeed != null){{
      const ySd = y + 2 + barH + gap;
      const xSd = xOf(r.vsSeed);
      const wSd = Math.max(2, Math.abs(xSd - x1));
      const bSd = el('rect', {{class:'bar-sd', x: Math.min(x1,xSd), y: ySd, width: wSd, height: barH, rx: 2}});
      bSd.dataset.idx = i; bSd.dataset.which = 'sd';
      svg.appendChild(bSd);
    }}
    // missing marker
    if (r.vsPy == null && r.vsSeed == null){{
      const m = el('text', {{class:'missing', x: x1 + 8, y: y + rowH/2 + 3}});
      m.textContent = 'missing final (timeout)';
      svg.appendChild(m);
    }}
    // row label
    const t = el('text', {{class: 'row-label' + (muted ? ' muted' : ''), x: leftPad - 10, y: y + rowH/2 + 3, 'text-anchor':'end'}});
    t.textContent = r.name;
    svg.appendChild(t);
  }});

  const ax = el('g', {{class:'axis'}});
  ticks.forEach(t => {{
    const x = xOf(t);
    const lab = el('text', {{x:x, y:H-botPad+16, 'text-anchor':'middle'}});
    lab.textContent = t < 1000 ? (t + '×') : (t/1000 + 'k×');
    ax.appendChild(lab);
  }});
  svg.appendChild(ax);

  const cap = el('text', {{class:'ylabel', x: leftPad - 10, y: topPad - 6, 'text-anchor':'end'}});
  cap.textContent = 'problem (sorted by vs-PyTorch)';
  svg.appendChild(cap);

  function fmtMs(v){{ return (v==null) ? 'n/a' : v.toFixed(3) + ' ms'; }}
  svg.addEventListener('mousemove', (e) => {{
    const target = e.target;
    if (target.tagName !== 'rect') {{ tt.style.opacity = 0; return; }}
    const i = +target.dataset.idx;
    const which = target.dataset.which;
    const r = rows[i];
    const cwRect = cw.getBoundingClientRect();
    const px = e.clientX - cwRect.left + 12;
    const py = e.clientY - cwRect.top + 12;
    tt.style.left = px + 'px';
    tt.style.top = py + 'px';
    const pyTxt = (r.vsPy==null) ? 'n/a' : r.vsPy.toFixed(2) + '×';
    const sdTxt = (r.vsSeed==null) ? 'n/a' : r.vsSeed.toFixed(2) + '×';
    tt.innerHTML = `<div class="t">${{r.name}}</div>` +
      `<div class="r"><span class="py">vs PyTorch: ${{pyTxt}}</span></div>` +
      `<div class="r"><span class="sd">vs seed: ${{sdTxt}}</span></div>` +
      `<div class="r">py: ${{fmtMs(r.baseMs)}} · seed: ${{fmtMs(r.seedMs)}} · champ: ${{fmtMs(r.champMs)}}</div>`;
    tt.style.opacity = 1;
  }});
  svg.addEventListener('mouseleave', () => {{ tt.style.opacity = 0; }});
}})();
</script>
</body>
</html>
"""
    OUTPUT.write_text(doc)
    print(OUTPUT)


if __name__ == "__main__":
    main()
