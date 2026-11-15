"""Layer knockout (research question 3).

    python src/analysis/knockout.py

Removes one part of a trained network at a time, at test time only, and measures how much test
accuracy drops. Probes show what a layer knows; knockout shows what the network depends on.

Details, fixed before looking at results:
* Every block is residual (output = input + branch), so "removing" means dropping the branch and
  keeping the skip path. For a ViT block there are three versions:
    - block:     the whole block (attention and MLP) becomes the identity;
    - attention: only the attention branch is dropped, the MLP still runs;
    - mlp:       only the MLP branch is dropped, attention still runs.
  For ResNet-18 only "block" applies: the residual branch is dropped and the shortcut kept, which
  is the identity for the 5 blocks without downsampling and the 1x1 projection for the 3 with it.
* Accuracy on all 10,000 test images, same evaluation transform as training; one knockout at a time.
* The intact model is re-evaluated first and must match the run's metrics.json.

Writes results/analysis/knockout.{json,md,png}.
"""
import argparse
import statistics

import torch

from common import (ALL_RUNS, MODELS, device_and_amp, finished_seeds, load_trained, make_loader, save_json,
                    save_text, style_axes, vit_layers)

MODES = ["block", "attention", "mlp"]


@torch.no_grad()
def accuracy(model, loader, device, amp):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def knockout_hook(model, mode, i):
    """Register the hook that removes part `mode` of block i; returns the handle."""
    if hasattr(model, "hf"):
        layer = vit_layers(model)[i]
        if mode == "block":
            def identity(m, args, kwargs, out):
                hidden = args[0] if args else kwargs["hidden_states"]
                return (hidden,) + tuple(out[1:]) if isinstance(out, tuple) else hidden
            return layer.register_forward_hook(identity, with_kwargs=True)
        if mode == "attention":
            def zero_attention(m, args, out):
                return (torch.zeros_like(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else torch.zeros_like(out)
            return layer.attention.register_forward_hook(zero_attention)
        if mode == "mlp":
            return layer.mlp.register_forward_hook(lambda m, args, out: torch.zeros_like(out))
    else:
        block = model.stages[i]
        if mode == "block":
            return block.register_forward_hook(lambda m, args, out: torch.relu(m.shortcut(args[0])))
    raise ValueError(f"mode {mode} does not apply to this model")


def n_blocks(model):
    return len(vit_layers(model)) if hasattr(model, "hf") else len(model.stages)


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"block": "Whole block removed", "attention": "Attention branch removed", "mlp": "MLP branch removed"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, mode in zip(axes, MODES):
        for run, r in results.items():
            if mode not in r["drops"]:
                continue
            rows = r["drops"][mode]
            depth = [(i + 1) / len(rows) for i in range(len(rows))]
            ax.errorbar(depth, [row["mean"] for row in rows], yerr=[row["std"] for row in rows],
                        color=MODELS[run][1], linewidth=1.6, marker="o", markersize=5, markeredgecolor="white",
                        markeredgewidth=0.8, elinewidth=1.0, capsize=0, label=MODELS[run][0], zorder=3)
            proj = r.get("projection_blocks", []) if mode == "block" else []
            if proj:
                ax.scatter([b / len(rows) for b in proj], [rows[b - 1]["mean"] for b in proj], s=42,
                           facecolors="white", edgecolors=MODELS[run][1], linewidths=1.6, zorder=4)
        ax.axhline(0, color="#8a8984", linewidth=1.0, zorder=2)
        ax.set_title(titles[mode], fontsize=10, color="#26251f", loc="left")
        ax.set_xlabel("Relative depth of the removed block", fontsize=9, color="#5d5c57")
        ax.set_xlim(0, 1.05)
        style_axes(ax)
    axes[0].set_ylabel("Test accuracy drop (points)", fontsize=9, color="#5d5c57")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=min(len(labels), 6),
               frameon=False, fontsize=9)
    fig.suptitle("What the network depends on: accuracy lost when one block is removed (mean ± std over seeds)",
                 fontsize=11, color="#26251f", x=0.01, ha="left")
    if any(r.get("projection_blocks") for r in results.values()):
        fig.text(0.01, 0.005, "Hollow markers: ResNet-18 blocks that downsample. Removing them keeps a 1x1 projection "
                 "shortcut instead of the identity, so they are not comparable with the other blocks.",
                 fontsize=8, color="#5d5c57", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=ALL_RUNS)
    p.add_argument("--n", type=int, default=None, help="test images (default: all 10,000)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    device, amp = device_and_amp()

    results = {}
    for run in args.runs:
        seeds = finished_seeds(run)
        if not seeds:
            print(f"skip {run}: no finished seeds")
            continue
        per_seed, base = {}, {}
        for seed, seed_dir in seeds:
            model, cfg, metrics = load_trained(seed_dir, device, attn="sdpa")
            loader = make_loader(cfg, "test", n=args.n, batch_size=256, workers=args.workers)
            acc0 = accuracy(model, loader, device, amp)
            base[seed] = acc0
            if args.n is None and abs(acc0 - metrics["final_test_acc"]) > 0.002:
                print(f"WARNING {run} seed {seed}: intact accuracy {acc0:.4f} differs from metrics.json "
                      f"{metrics['final_test_acc']:.4f}")
            modes = MODES if hasattr(model, "hf") else ["block"]
            per_seed[seed] = {}
            for mode in modes:
                drops = []
                for i in range(n_blocks(model)):
                    handle = knockout_hook(model, mode, i)
                    try:
                        drops.append(100 * (acc0 - accuracy(model, loader, device, amp)))
                    finally:
                        handle.remove()
                per_seed[seed][mode] = drops
            print(f"{run} seed {seed}: intact {100 * acc0:.2f}%; " + "; ".join(
                f"{mode} drops " + " ".join(f"{d:.1f}" for d in drops) for mode, drops in per_seed[seed].items()))
            del model
        drops = {}
        for mode in per_seed[next(iter(per_seed))]:
            k = len(per_seed[next(iter(per_seed))][mode])
            drops[mode] = []
            for i in range(k):
                vals = [per_seed[s][mode][i] for s in per_seed]
                drops[mode].append({"per_seed": vals, "mean": statistics.mean(vals),
                                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0})
        results[run] = {"label": MODELS[run][0], "intact_acc": base, "drops": drops}
        if run == "resnet18":
            model_blocks = load_trained(seeds[0][1], torch.device("cpu"), attn="sdpa")[0].stages
            results[run]["projection_blocks"] = [i + 1 for i, b in enumerate(model_blocks) if len(b.shortcut) > 0]

    if not results:
        print("nothing to analyse")
        return
    save_json("knockout.json", {"n_test": args.n or 10000, "units": "accuracy drop in points", "models": results})
    lines = ["# Layer knockout\n",
             "Test accuracy drop (points) when one block's branch is removed at test time, mean ± std over "
             "seeds. Block numbers start at 1. Generated by `python src/analysis/knockout.py`.\n",
             "For ResNet-18, blocks marked * downsample: removing them keeps a 1x1 projection shortcut instead of "
             "the identity, so they are not comparable with the other blocks.\n",
             "| Model | Intact accuracy (%) | Removed | Block | Accuracy drop (points) |",
             "| --- | --- | --- | --- | --- |"]
    for run, r in results.items():
        intact = statistics.mean(r["intact_acc"].values()) * 100
        for mode, rows in r["drops"].items():
            for i, row in enumerate(rows):
                mark = "*" if i + 1 in r.get("projection_blocks", []) else ""
                lines.append(f"| {r['label']} | {intact:.2f} | {mode} | {i + 1}{mark} | {row['mean']:.2f} ± {row['std']:.2f} |")
    save_text("knockout.md", "\n".join(lines) + "\n")
    plot(results, "results/analysis/knockout.png")
    print("wrote results/analysis/knockout.{json,md,png}")


if __name__ == "__main__":
    main()
