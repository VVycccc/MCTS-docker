#!/usr/bin/env python3
"""Generate a horizontal grouped bar chart with a log-scale speedup axis."""

from __future__ import annotations

import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "MCTS_28_SPEEDUPS.md"
OUTPUT = ROOT / "figures" / "mcts_28_speedups_bar_log.svg"

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
        rows.append({"level": section, "name": name, "seed": parse_speedup(seed), "pytorch": parse_speedup(pytorch)})
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
    width = 1280
    left = 235
    right = 84
    top = 96
    bottom = 88
    row_h = 32
    gap_after_l1 = 22
    plot_w = width - left - right
    height = top + bottom + row_h * len(rows) + gap_after_l1

    min_x = 0.08
    max_x = 220.0
    log_min = math.log10(min_x)
    log_max = math.log10(max_x)

    def x_pos(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * plot_w

    x_one = x_pos(1.0)
    tick_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    major_ticks = {0.1, 1, 10, 100}

    y_positions = []
    y = top
    for idx, _row in enumerate(rows):
        y_positions.append(y)
        y += row_h
        if idx == 17:
            y += gap_after_l1

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    out.append(f'<rect width="100%" height="100%" fill="{SURFACE}"/>')
    out.append('<style>')
    out.append('text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:23px;font-weight:700}.subtitle{font-size:13px;fill:#64748b}.label{font-size:13px}.tick{font-size:11px;fill:#64748b}.legend{font-size:12px;fill:#334155}.value{font-size:10.5px;fill:#475569}.level{font-size:12px;font-weight:700;fill:#64748b;letter-spacing:.04em}.missing{font-size:11px;fill:#9ca3af}')
    out.append('</style>')

    out.append(f'<text x="{left}" y="34" class="title">DirecTune-MCTS speedups across 28 problems</text>')
    out.append(f'<text x="{left}" y="58" class="subtitle">Horizontal grouped bars on a log-scale speedup axis. Bars grow right of 1× for speedup and left of 1× for regression.</text>')

    legend_x = width - 360
    legend_y = 34
    out.append(f'<rect x="{legend_x}" y="{legend_y - 13}" width="28" height="10" rx="3" fill="{BLUE}"/>')
    out.append(f'<text x="{legend_x + 38}" y="{legend_y - 4}" class="legend">vs naive seed</text>')
    out.append(f'<rect x="{legend_x + 162}" y="{legend_y - 13}" width="28" height="10" rx="3" fill="{ORANGE}"/>')
    out.append(f'<text x="{legend_x + 200}" y="{legend_y - 4}" class="legend">vs PyTorch</text>')

    plot_top = top - 20
    plot_bottom = height - bottom + 13
    for tick in tick_values:
        x = x_pos(tick)
        stroke = "#cbd5e1" if tick == 1 else GRID
        sw = "1.7" if tick == 1 else "1"
        opacity = "1" if tick in major_ticks or tick == 1 else "0.55"
        out.append(f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')
        out.append(f'<text x="{x:.2f}" y="{plot_bottom + 23}" text-anchor="middle" class="tick">{tick:g}×</text>')

    out.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" class="subtitle">Speedup factor, log scale</text>')
    out.append(f'<text x="{x_one + 6:.2f}" y="{plot_top - 8}" class="tick">1× baseline</text>')

    out.append(f'<text x="42" y="{y_positions[0] - 12}" class="level">L1</text>')
    out.append(f'<text x="42" y="{y_positions[18] - 12}" class="level">L2</text>')
    divider_y = (y_positions[17] + y_positions[18]) / 2 - 3
    out.append(f'<line x1="35" y1="{divider_y:.2f}" x2="{width - right}" y2="{divider_y:.2f}" stroke="#d1d5db" stroke-dasharray="4 5"/>')

    bar_h = 9
    for idx, (row, yy) in enumerate(zip(rows, y_positions)):
        if idx % 2 == 0:
            out.append(f'<rect x="32" y="{yy - 15}" width="{width - 96}" height="{row_h}" fill="#f8fafc" opacity="0.65" rx="5"/>')
        out.append(f'<text x="42" y="{yy + 4}" class="label">{escape(str(row["name"]))}</text>')

        for key, color, dy in [("seed", BLUE, -6), ("pytorch", ORANGE, 6)]:
            value = row[key]
            if value is None:
                continue
            xv = x_pos(float(value))
            x = min(x_one, xv)
            w = abs(xv - x_one)
            # Keep exact-1x bars visible while still anchored to the 1x reference line.
            if w < 3:
                w = 3
                x = x_one - 1.5
            out.append(f'<rect x="{x:.2f}" y="{yy + dy - bar_h / 2:.2f}" width="{w:.2f}" height="{bar_h}" rx="4" fill="{color}"/>')

            show_label = float(value) >= 5 or float(value) < 0.5 or row["name"] in {"18_Matmul", "40_layernorm", "90_cumprod", "57_convT2d", "97_sdpa"}
            if show_label:
                if float(value) >= 1:
                    tx = min(xv + 6, width - right + 35)
                    anchor = "start"
                else:
                    tx = max(xv - 6, left - 35)
                    anchor = "end"
                out.append(f'<text x="{tx:.2f}" y="{yy + dy + 3}" text-anchor="{anchor}" class="value">{fmt(float(value))}</text>')

        if row["seed"] is None and row["pytorch"] is None:
            out.append(f'<text x="{x_one + 8:.2f}" y="{yy + 4}" class="missing">missing final</text>')

    out.append(f'<text x="{width - right}" y="{height - 18}" text-anchor="end" class="subtitle">Data: MCTS_28_SPEEDUPS.md</text>')
    out.append('</svg>')
    OUTPUT.write_text("\n".join(out) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
