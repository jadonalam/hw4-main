"""
scripts/plot_runs.py  —  Generate report-ready plots from training outputs.

Usage:
    python scripts/plot_runs.py
    python scripts/plot_runs.py --watch --interval 30
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def save_loss_curve(
    series: list[np.ndarray | list[float]],
    labels: list[str],
    out: str | Path,
    title: str,
):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays = [np.asarray(values, dtype=float) for values in series if len(values) > 0]
    labels = [label for values, label in zip(series, labels) if len(values) > 0]
    if not arrays:
        return

    width, height = 1000, 620
    left, right, top, bottom = 85, 35, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2 - len(title) * 3, 18), title, fill="black")
    draw.text((width // 2 - 20, height - 35), "epoch", fill="black")
    draw.text((18, height // 2 - 10), "loss", fill="black")

    max_len = max(len(values) for values in arrays)
    y_min = min(float(values.min()) for values in arrays)
    y_max = max(float(values.max()) for values in arrays)
    padding = 0.05 * max(y_max - y_min, 1e-6)
    y_min -= padding
    y_max += padding

    draw.line([(left, top), (left, top + plot_h), (left + plot_w, top + plot_h)], fill="black", width=2)

    for frac in np.linspace(0, 1, 6):
        y_value = y_min + frac * (y_max - y_min)
        y = top + plot_h - frac * plot_h
        draw.line([(left - 5, y), (left + plot_w, y)], fill="#dddddd")
        draw.text((left - 75, y - 7), f"{y_value:.4f}", fill="black")

    for frac in np.linspace(0, 1, 6):
        epoch = 1 + frac * max(max_len - 1, 1)
        x = left + frac * plot_w
        draw.line([(x, top + plot_h), (x, top + plot_h + 5)], fill="black")
        draw.text((x - 10, top + plot_h + 12), f"{int(round(epoch))}", fill="black")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for idx, values in enumerate(arrays):
        if len(values) == 1:
            x_vals = np.array([left], dtype=float)
        else:
            x_vals = left + np.arange(len(values)) / (len(values) - 1) * plot_w
        y_vals = top + (y_max - values) / (y_max - y_min) * plot_h
        points = list(zip(x_vals.astype(float), y_vals.astype(float)))
        color = colors[idx % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = left + 15 + idx * 130
        draw.line([(legend_x, top + 15), (legend_x + 25, top + 15)], fill=color, width=3)
        draw.text((legend_x + 32, top + 8), labels[idx], fill="black")

    image.save(out)


def plot_all(vp_dir: Path, rf_dir: Path, reflow_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    vp_train = _load(vp_dir / "train_losses.npy")
    vp_val = _load(vp_dir / "val_losses.npy")
    if vp_train is not None:
        series = [vp_train]
        labels = ["train"]
        if vp_val is not None:
            series.append(vp_val)
            labels.append("val")
        save_loss_curve(series, labels, out_dir / "vp_loss_curve.png", "VP score model loss")

    rf_train = _load(rf_dir / "train_losses.npy")
    if rf_train is not None:
        save_loss_curve([rf_train], ["train"], out_dir / "rectflow_loss_curve.png", "Rectified flow loss")

    reflow_train = _load(reflow_dir / "train_losses.npy")
    if reflow_train is not None:
        save_loss_curve([reflow_train], ["train"], out_dir / "reflow_loss_curve.png", "Reflow retraining loss")

    print(f"Updated plots in {out_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vp_dir", type=Path, default=Path("runs/vp"))
    parser.add_argument("--rf_dir", type=Path, default=Path("runs/rectflow"))
    parser.add_argument("--reflow_dir", type=Path, default=Path("runs/rectflow_reflow"))
    parser.add_argument("--out_dir", type=Path, default=Path("runs/plots"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = get_args()
    while True:
        plot_all(args.vp_dir, args.rf_dir, args.reflow_dir, args.out_dir)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
