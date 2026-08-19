#!/usr/bin/env python3
"""Generate an SVG log-scale speedup plot from MCTS_28_SPEEDUPS.md."""

from __future__ import annotations

import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "MCTS_28_SPEEDUPS.md"
OUTPUT = ROOT / "figures" / "mcts_28_speedups_log.svg"

BLUE = "#0072B2"      # vs seed
ORANGE = "#D55E00"    # vs PyTorch
INK = "#1f2933"
MUTED = "#64748b"
GRID = "#e5e7eb"
SURFACE = "#fcfcfb"
MISSING = "#9ca3af"

ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")


def parse_speedup(value: str) -> float | None:
    value = value.strip()
    if value == "—":
        return None
    return float(value.replace("×", "").strip())


def parse_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    section = None
    for line in INPUT.read_text().splitlines():
        if line.startswith("## L1"):
            section = "L1"
            continue
        if line.startswith("## L2"):
            section = "L2"
            continue
        if not section or not line.startswith("|") or line.startswith("|---"):
            continue
        match = ROW_RE.match(line)
        if not match:
            continue
        name, seed, pytorch = [x.strip() for x in match.groups()]
        if name == "题目":
            continue
        rows.append(
            {
                "level": section,
                "name": name,
                "seed": parse_speedup(seed),
                "pytorch": parse_speedup(pytorch),
            }
        )
    return rows


def fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 100:
        return f"{value:.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def main() -> None:
    rows = parse_rows()
    if not rows:
        raise SystemExit(f"No rows parsed from {INPUT}")

    width = 1180
    left = 230
    right = 70
    top = 92
    bottom = 82
    row_h = 28
    gap_after_l1 = 18
    plot_w = width - left - right
    height = top + bottom + row_h * len(rows) + gap_after_l1

    min_x = 0.08
    max_x = 220.0
    log_min = math.log10(min_x)
    log_max = math.log10(max_x)

    def x_pos(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * plot_w

    y_positions = []
    y = top
    for idx, row in enumerate(rows):
        y_positions.append(y)
        y += row_h
        if idx == 17:
            y += gap_after_l1

    tick_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    major_ticks = {0.1, 1, 10, 100}

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append(f'<rect width="100%" height="100%" fill="{SURFACE}"/>')
    out.append('<style>')
    out.append('text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1f2933} .small{font-size:12px}.label{font-size:13px}.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#64748b}.level{font-size:12px;font-weight:700;fill:#64748b;letter-spacing:.04em}.tick{font-size:11px;fill:#64748b}.value{font-size:10.5px;fill:#475569}.missing{font-size:11px;fill:#9ca3af}.legend{font-size:12px;fill:#334155}')
    out.append('</style>')

    out.append(f'<text x="{left}" y="34" class="title">DirecTune-MCTS speedups across 28 problems</text>')
    out.append(f'<text x="{left}" y="58" class="subtitle">Log-scale x-axis; vertical reference line marks 1×. Missing points indicate no final champion.</text>')

    legend_y = 34
    legend_x = width - 330
    out.append(f'<circle cx="{legend_x}" cy="{legend_y - 4}" r="5" fill="{BLUE}"/>')
    out.append(f'<text x="{legend_x + 12}" y="{legend_y}" class="legend">vs naive seed</text>')
    out.append(f'<circle cx="{legend_x + 145}" cy="{legend_y - 4}" r="5" fill="{ORANGE}"/>')
    out.append(f'<text x="{legend_x + 157}" y="{legend_y}" class="legend">vs PyTorch</text>')

    plot_top = top - 18
    plot_bottom = height - bottom + 12
    for tick in tick_values:
        x = x_pos(tick)
        stroke = "#cbd5e1" if tick == 1 else GRID
        opacity = "1" if tick in major_ticks or tick == 1 else "0.55"
        sw = "1.5" if tick == 1 else "1"
        out.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')
        if tick in major_ticks or tick in {0.2, 0.5, 2, 5, 20, 50, 200}:
            out.append(f'<text x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="middle" class="tick">{tick:g}×</text>')

    out.append(f'<text x="{left}" y="{plot_bottom + 52}" text-anchor="middle" class="tick">slower</text>')
    out.append(f'<text x="{x_pos(1):.2f}" y="{plot_bottom + 52}" text-anchor="middle" class="tick">1×</text>')
    out.append(f'<text x="{left + plot_w}" y="{plot_bottom + 52}" text-anchor="middle" class="tick">faster</text>')

    out.append(f'<text x="40" y="{y_positions[0] - 11}" class="level">L1</text>')
    out.append(f'<text x="40" y="{y_positions[18] - 11}" class="level">L2</text>')
    divider_y = (y_positions[17] + y_positions[18]) / 2 - 3
    out.append(f'<line x1="35" y1="{divider_y:.2f}" x2="{width - right}" y2="{divider_y:.2f}" stroke="#d1d5db" stroke-dasharray="4 5"/>')

    for idx, (row, yy) in enumerate(zip(rows, y_positions)):
        if idx % 2 == 0:
            out.append(f'<rect x="32" y="{yy - 14}" width="{width - 92}" height="{row_h}" fill="#f8fafc" opacity="0.55" rx="5"/>')
        name = escape(str(row["name"]))
        out.append(f'<text x="42" y="{yy + 4}" text-anchor="start" class="label">{name}</text>')

        seed = row["seed"]
        pytorch = row["pytorch"]
        values = [("seed", seed, BLUE, -5), ("pytorch", pytorch, ORANGE, 5)]
        numeric = [(v, color, dy) for _, v, color, dy in values if isinstance(v, float)]
        if len(numeric) == 2:
            x1 = x_pos(numeric[0][0])
            x2 = x_pos(numeric[1][0])
            out.append(f'<line x1="{x1:.2f}" y1="{yy}" x2="{x2:.2f}" y2="{yy}" stroke="#94a3b8" stroke-width="1.2" opacity="0.55"/>')
        for _, value, color, dy in values:
            if value is None:
                continue
            x = x_pos(value)
            out.append(f'<circle cx="{x:.2f}" cy="{yy + dy}" r="5.2" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
            label_anchor = "start"
            label_x = x + 8
            if value > 50:
                label_anchor = "end"
                label_x = x - 8
            show_label = value >= 10 or value < 0.5 or row["name"] in {"18_Matmul", "40_layernorm", "97_sdpa", "57_convT2d"}
            if show_label:
                out.append(f'<text x="{label_x:.2f}" y="{yy + dy + 3}" text-anchor="{label_anchor}" class="value">{fmt(value)}</text>')
        if seed is None and pytorch is None:
            out.append(f'<text x="{x_pos(1) + 8:.2f}" y="{yy + 4}" class="missing">missing final</text>')

    out.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" class="subtitle">Speedup factor, log scale</text>')
    out.append(f'<text x="{width - right}" y="{height - 18}" text-anchor="end" class="subtitle">Data: MCTS_28_SPEEDUPS.md</text>')
    out.append('</svg>')

    OUTPUT.write_text("\n".join(out) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
