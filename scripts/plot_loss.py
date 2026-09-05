"""Plot training loss curve from logs/full.log -> public/assets/loss-curve.svg.

Stdlib only. Log-scale y-axis (losses span 1.353 -> 0.000).

Usage:
    /usr/bin/python3 scripts/plot_loss.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "full.log"
OUT = ROOT / "public" / "assets" / "loss-curve.svg"

W, H = 760, 420
PAD_L, PAD_R, PAD_T, PAD_B = 64, 20, 24, 48
Y_MIN, Y_MAX = -3.5, 0.3  # log10 range: ~0.0003 .. ~2.0


def parse():
    text = LOG.read_text()
    train = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Train loss ([0-9.]+)", text)]
    val = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", text)]
    return train, val


def x_of(it, lo, hi):
    return PAD_L + (it - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def y_of(loss):
    import math
    v = math.log10(max(loss, 1e-4))
    return PAD_T + (Y_MAX - v) / (Y_MAX - Y_MIN) * (H - PAD_T - PAD_B)


def main():
    import math
    train, val = parse()
    lo = min(i for i, _ in train)
    hi = max(i for i, _ in train)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img" aria-label="Wamda-3B training loss curve">']
    parts.append('<rect width="100%" height="100%" fill="#0d1117" rx="12"/>')
    parts.append('<text x="24" y="30" fill="#e6edf3" font-family="sans-serif" font-size="16" font-weight="bold">Wamda-3B — training loss (log scale)</text>')

    # gridlines + y labels
    for exp in range(-3, 1):
        y = y_of(10 ** exp)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="#21262d" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" fill="#8b949e" font-family="monospace" font-size="11" text-anchor="end">1e{exp}</text>')

    # x labels
    for it in range(0, hi + 1, 100):
        x = x_of(it, lo, hi)
        parts.append(f'<text x="{x:.1f}" y="{H - 18}" fill="#8b949e" font-family="monospace" font-size="11" text-anchor="middle">iter {it}</text>')

    # train polyline (skip loss <= 0 for log scale)
    pts = [(x_of(i, lo, hi), y_of(v)) for i, v in train if v > 0]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    parts.append(f'<path d="{d}" fill="none" stroke="#f0b429" stroke-width="2.5"/>')

    # val markers
    for i, v in val:
        if v <= 0:
            continue
        x, y = x_of(i, lo, hi), y_of(v)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#1f6feb" stroke="#e6edf3" stroke-width="1.5"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" fill="#79c0ff" font-family="monospace" font-size="11" text-anchor="middle">{v}</text>')

    # legend
    parts.append(f'<line x1="{W - 250}" y1="28" x2="{W - 220}" y2="28" stroke="#f0b429" stroke-width="2.5"/>')
    parts.append(f'<text x="{W - 214}" y="32" fill="#8b949e" font-family="sans-serif" font-size="12">train</text>')
    parts.append(f'<circle cx="{W - 150}" cy="28" r="5" fill="#1f6feb" stroke="#e6edf3" stroke-width="1.5"/>')
    parts.append(f'<text x="{W - 140}" y="32" fill="#8b949e" font-family="sans-serif" font-size="12">val</text>')

    parts.append('</svg>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT} ({len(train)} train pts, {len(val)} val pts)")


if __name__ == "__main__":
    main()
