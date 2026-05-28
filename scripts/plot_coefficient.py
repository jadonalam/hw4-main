"""
scripts/plot_coefficient.py  —  Part 1.8
=========================================
Plot the DDPM loss coefficient
    β_t² / (2 σ_t² α_t (1 - ᾱ_t))
vs. t on a log-scale y-axis.

Usage::
    python scripts/plot_coefficient.py --T 1000 --beta_start 1e-4 --beta_end 0.02
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw


def linear_schedule(T: int, beta_start: float, beta_end: float):
    return np.linspace(beta_start, beta_end, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",          type=int,   default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end",   type=float, default=0.02)
    parser.add_argument("--out",        type=str,   default="coefficient_plot.png")
    args = parser.parse_args()

    betas = linear_schedule(args.T, args.beta_start, args.beta_end)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    sigma_sq = betas
    coefficient = betas**2 / (2.0 * sigma_sq * alphas * (1.0 - alpha_bars))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    timesteps = np.arange(1, args.T + 1)

    width, height = 1050, 650
    left, right, top, bottom = 90, 35, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    log_coeff = np.log10(coefficient)
    y_min = float(np.floor(log_coeff.min()))
    y_max = float(np.ceil(log_coeff.max()))
    if y_min == y_max:
        y_max += 1.0

    draw.line([(left, top), (left, top + plot_h), (left + plot_w, top + plot_h)], fill="black", width=2)
    draw.text((width // 2 - 95, 15), "DDPM Loss Coefficient", fill="black")
    draw.text((width // 2 - 5, height - 30), "t", fill="black")
    draw.text((8, height // 2 - 20), "log10 coefficient", fill="black")

    for frac in np.linspace(0, 1, 6):
        y_value = y_min + frac * (y_max - y_min)
        y = top + plot_h - frac * plot_h
        draw.line([(left - 5, y), (left + plot_w, y)], fill="#dddddd")
        draw.text((left - 70, y - 7), f"1e{int(round(y_value))}", fill="black")

    for frac in np.linspace(0, 1, 6):
        t_value = 1 + frac * (args.T - 1)
        x = left + frac * plot_w
        draw.line([(x, top + plot_h), (x, top + plot_h + 5)], fill="black")
        draw.text((x - 12, top + plot_h + 12), f"{int(round(t_value))}", fill="black")

    x_vals = left + (timesteps - 1) / max(args.T - 1, 1) * plot_w
    y_vals = top + (y_max - log_coeff) / (y_max - y_min) * plot_h
    points = list(zip(x_vals.astype(float), y_vals.astype(float)))
    draw.line(points, fill="#1f77b4", width=3)

    image.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
