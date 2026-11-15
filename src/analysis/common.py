"""Shared helpers for the layer analyses (research question 3).

Every analysis loads the final checkpoint of a finished run (results/<run>/seed<k>/last.pt),
rebuilds the model with eager attention so attention maps can be read, and evaluates it on
CIFAR-10 with the same evaluation transform the run was trained with. Nothing here trains.
"""
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path.insert(0, str(SRC))
from data import TensorImages, build_transform, load_split  # noqa: E402
from models import build_model  # noqa: E402

RESULTS = Path("results")
OUT_DIR = RESULTS / "analysis"
CIFAR_PX = 32  # original CIFAR-10 width; distances are reported in these pixels

# One fixed label and color per model, used in every figure (validated categorical palette, fixed order).
MODELS = {
    "resnet18": ("ResNet-18", "#2a78d6"),
    "vitlite_p8": ("ViT-Lite-7/8", "#eb6834"),
    "vitlite_p4": ("ViT-Lite-7/4", "#1baf7a"),
    "vitlite_p2": ("ViT-Lite-7/2", "#eda100"),
    "vitlite_convstem": ("ViT-Lite-7 + conv stem", "#e87ba4"),
    "vitb16_finetune": ("ViT-B/16, pretrained + fine-tuned", "#008300"),
    "vitb16_in21k": ("ViT-B/16, pretrained only", "#6250d6"),
}
VIT_RUNS = ["vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem", "vitb16_finetune"]
ALL_RUNS = ["resnet18"] + VIT_RUNS
PRETRAINED_ONLY = "vitb16_in21k"


def device_and_amp():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        return dev, (torch.bfloat16 if torch.cuda.is_bf16_supported() else None)
    return torch.device("cpu"), None


def finished_seeds(run):
    """Seed directories of `run` that have both a final checkpoint and final metrics."""
    out = []
    for seed_dir in sorted((RESULTS / run).glob("seed*")):
        if (seed_dir / "last.pt").exists() and (seed_dir / "metrics.json").exists():
            out.append((int(seed_dir.name[4:]), seed_dir))
    return out


def load_trained(seed_dir, device, attn="eager"):
    """Rebuild a trained model from its run directory. Returns (model, config dict, metrics dict)."""
    cfg = yaml.safe_load((seed_dir / "config.yaml").read_text())
    mcfg = dict(cfg["model"])
    mcfg["attn"] = attn
    if mcfg["name"] == "vit_b16_pretrained":
        # Same architecture; the fine-tuned weights come from the checkpoint, so nothing is downloaded.
        mcfg = {"name": "vit_b16_architecture", "attn": attn}
    model = build_model(mcfg)
    ck = torch.load(seed_dir / "last.pt", map_location="cpu")
    model.load_state_dict(ck["model"])
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    return model.to(device).eval(), cfg, metrics


def load_pretrained_only(device, attn="eager"):
    """The ImageNet-21k checkpoint before any CIFAR-10 training (its 10-class head is random and unused).

    Needs the checkpoint in the Hugging Face cache (experiments/prepare_data.py --cache-vit)."""
    cfg = yaml.safe_load((REPO / "experiments/configs/vitb16_finetune.yaml").read_text())
    model = build_model({"name": "vit_b16_pretrained", "attn": attn})
    return model.to(device).eval(), cfg


def make_loader(cfg, split, n=None, batch_size=256, workers=4):
    """Evaluation-mode loader (no augmentation) for the first n images of a split."""
    x, y = load_split(cfg["data"]["dir"], split)
    if n:
        x, y = x[:n], y[:n]
    ds = TensorImages(x, y, build_transform(cfg["data"], train=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def vit_layers(model):
    vit = model.hf.vit
    return vit.layers if hasattr(vit, "layers") else vit.encoder.layer


def image_size(cfg):
    return int(cfg["data"].get("image_size", 32))


def token_grid_distances(n_patch_tokens, img_size):
    """Pairwise distances between patch-token centres, as a fraction of the image width.

    Works for any square token grid: 4x4, 8x8 and 16x16 patches on 32 px, the conv stem's 8x8 grid,
    and ViT-B/16's 14x14 grid on 224 px."""
    g = int(round(math.sqrt(n_patch_tokens)))
    if g * g != n_patch_tokens:
        raise ValueError(f"{n_patch_tokens} patch tokens is not a square grid")
    cell = img_size / g
    idx = torch.arange(g, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    centres = torch.stack([yy.flatten(), xx.flatten()], dim=1) * cell + cell / 2
    return torch.cdist(centres, centres) / img_size


def save_json(name, obj):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(obj, indent=1))


def save_text(name, text):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(text)


def style_axes(ax):
    """Recessive grid and axes: light horizontal grid, no top/right spines."""
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors="#5d5c57", labelsize=9)
