"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--batch_size", type=int,  default=128)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(
    samples: torch.Tensor,
    directory: str,
    start_idx: int = 0,
    from_minus_one_to_one: bool = True,
):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = samples.detach().cpu()
    if from_minus_one_to_one:
        samples = samples.clamp(-1, 1) * 0.5 + 0.5
    else:
        samples = samples.clamp(0, 1)
    if samples.ndim == 4 and samples.size(1) == 1:
        samples = samples.repeat(1, 3, 1, 1)
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{start_idx + i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def load_state_dict(checkpoint: str, device: torch.device) -> dict:
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def load_vp_model(args, device: torch.device) -> tuple[VPSDE, UNet]:
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(load_state_dict(args.vp_checkpoint, device))
    model.eval()
    return sde, model


def load_rf_model(args, device: torch.device) -> tuple[RectifiedFlow, UNet]:
    flow = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(load_state_dict(args.rf_checkpoint, device))
    model.eval()
    return flow, model


@torch.no_grad()
def ddim_sample(
    sde: VPSDE,
    score_model: torch.nn.Module,
    shape: tuple[int, ...],
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    score_model.eval()
    batch_size = shape[0]
    view_shape = (batch_size, *([1] * (len(shape) - 1)))
    x = sde.sigma(torch.ones((), device=device)) * torch.randn(shape, device=device)
    times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t = torch.full((batch_size,), times[i].item(), device=device)
        t_next = torch.full((batch_size,), times[i + 1].item(), device=device)
        c_t = sde.c(t).reshape(view_shape).clamp_min(1e-5)
        sigma_t = sde.sigma(t).reshape(view_shape)
        c_next = sde.c(t_next).reshape(view_shape)
        sigma_next = sde.sigma(t_next).reshape(view_shape)

        score = score_model(x, t)
        eps = -sigma_t * score
        x0_hat = (x - sigma_t * eps) / c_t
        x = c_next * x0_hat + sigma_next * eps

    return x.clamp(-1, 1)


def save_real_fashionmnist(directory: str, n_samples: int):
    dataset = datasets.FashionMNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    for start in range(0, n_samples, 256):
        batch = []
        for offset in range(min(256, n_samples - start)):
            image, _ = dataset[(start + offset) % len(dataset)]
            batch.append(image)
        save_samples_to_dir(
            torch.stack(batch, dim=0),
            directory,
            start_idx=start,
            from_minus_one_to_one=False,
        )


def generate_samples_to_dir(
    method: str,
    num_steps: int,
    generated_dir: str,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    sde: VPSDE,
    vp_model: UNet,
    flow: RectifiedFlow,
    rf_model: UNet,
):
    written = 0
    while written < n_samples:
        current = min(batch_size, n_samples - written)
        shape = (current, 1, 28, 28)
        if method == "rectflow":
            samples = flow.euler_sample(rf_model, shape, num_steps=num_steps, device=device)
        elif method == "ddim":
            samples = ddim_sample(sde, vp_model, shape, num_steps=num_steps, device=device)
        elif method == "em":
            samples = sde.euler_maruyama(vp_model, shape, num_steps=num_steps, device=device)
        else:
            raise ValueError(f"Unknown method: {method}")
        save_samples_to_dir(samples, generated_dir, start_idx=written)
        written += current


def main():
    args = get_args()
    device = torch.device(args.device)

    sde, vp_model = load_vp_model(args, device)
    flow, rf_model = load_rf_model(args, device)

    with tempfile.TemporaryDirectory() as tmp_root:
        real_dir = os.path.join(tmp_root, "real")
        save_real_fashionmnist(real_dir, args.n_samples)

        rows = []
        for method in METHODS:
            for num_steps in STEP_COUNTS:
                generated_dir = os.path.join(tmp_root, f"{method}_{num_steps}")
                generate_samples_to_dir(
                    method,
                    num_steps,
                    generated_dir,
                    args.n_samples,
                    args.batch_size,
                    device,
                    sde,
                    vp_model,
                    flow,
                    rf_model,
                )
                metrics = compute_kid(generated_dir, real_dir)
                mean = metrics["kernel_inception_distance_mean"]
                std = metrics["kernel_inception_distance_std"]
                rows.append((method, num_steps, mean, std))
                print(f"{method:8s} | {num_steps:4d} steps | KID {mean:.6f} ± {std:.6f}")

        print("\n| Method | Steps | KID mean | KID std |")
        print("|---|---:|---:|---:|")
        for method, num_steps, mean, std in rows:
            print(f"| {method} | {num_steps} | {mean:.6f} | {std:.6f} |")


if __name__ == "__main__":
    main()
