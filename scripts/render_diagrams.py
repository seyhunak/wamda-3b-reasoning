"""Render PNG versions of README diagrams (PNG renders in every markdown viewer).

- loss-curve.png : from logs/full.log (same data as loss-curve.svg)
- pipeline.png   : static pipeline diagram (same content as pipeline.svg)

Usage:
    ./.venv/bin/python scripts/render_diagrams.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "public" / "assets"

BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"


def render_loss_curve():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    text = (ROOT / "logs" / "full.log").read_text()
    train = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Train loss ([0-9.]+)", text)]
    val = [(int(i), float(v)) for i, v in re.findall(r"Iter (\d+): Val loss ([0-9.]+)", text)]

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.semilogy([i for i, _ in train if _ > 0], [v for _, v in train if v > 0],
                color="#f0b429", linewidth=2, label="train")
    ax.semilogy([i for i, _ in val], [v for _, v in val],
                color="#1f6feb", marker="o", markersize=5,
                markerfacecolor="#1f6feb", markeredgecolor=FG, linestyle="None", label="val")
    for i, v in val:
        ax.annotate(f"{v:g}", (i, v), color="#79c0ff", fontsize=8,
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("iteration", color=MUTED)
    ax.set_ylabel("loss (log scale)", color=MUTED)
    ax.set_title("Wamda-3B — training loss", color=FG, fontweight="bold", loc="left")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#21262d")
    ax.legend(facecolor="#161b22", edgecolor="#21262d", labelcolor=FG)
    fig.tight_layout()
    out = OUT_DIR / "loss-curve.png"
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def render_pipeline():
    from PIL import Image, ImageDraw, ImageFont

    W, H, S = 760, 210, 2  # S = supersample for crisp text
    img = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16 * S)
    bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12 * S)
    reg = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11 * S)

    def rr(box, color):
        d.rounded_rectangle([c * S for c in box], radius=10 * S, outline=color, width=2)

    def ct(x, y, s, font, color):
        d.text((x * S, y * S), s, font=font, fill=color, anchor="mm")

    d.text((24 * S, 22 * S), "Wamda-3B pipeline", font=title, fill=FG)

    boxes = [
        ("#f0b429", ["1 · Dataset", "300 train / 40 valid", "verified AR traces", "build_dataset.py"]),
        ("#1f6feb", ["2 · LoRA SFT", "Qwen2.5-3B-Ins", "16 layers · 6.65M", "MLX · mask-prompt"]),
        ("#3fb950", ["3 · Adapter", "26.6 MB", "600 iters · ckpts/100", "adapters/"]),
        ("#a371f7", ["4 · Eval", "13/15 = 86.7%", "base 11/15 = 73.3%", "eval/"]),
    ]
    xs = [24, 206, 388, 570]
    ws = [150, 150, 150, 166]
    for x, w, (color, lines) in zip(xs, ws, boxes):
        rr((x, 60, x + w, 170), color)
        ct(x + w / 2, 82, lines[0], bold, color)
        ct(x + w / 2, 102, lines[1], reg, FG)
        ct(x + w / 2, 120, lines[2], reg, MUTED)
        ct(x + w / 2, 138, lines[3], reg, MUTED)
    for x0, x1 in [(174, 206), (356, 388), (538, 570)]:
        y = 115 * S
        d.line([(x0 * S, y), (x1 * S, y)], fill=MUTED, width=2)
        d.polygon([(x1 * S, y), ((x1 - 8) * S, (y - 5 * S)), ((x1 - 8) * S, (y + 5 * S))], fill=MUTED)

    img = img.resize((W, H), Image.LANCZOS)
    out = OUT_DIR / "pipeline.png"
    img.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_loss_curve()
    render_pipeline()
