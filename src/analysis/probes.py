"""Linear probes on every layer (research question 3).

    python src/analysis/probes.py --n-train 10000 --n-test 10000

Freezes each trained model, takes its features after every block, and fits a logistic regression
to predict the CIFAR-10 class. Test accuracy per layer shows where class information appears
inside the network (Raghu et al., 2021, use the same tool to compare ViTs and ResNets).

Details, fixed before looking at results:
* ViTs: two feature types per layer, the CLS token and the mean of the patch tokens. Layer 0 is
  the embedding output (before any block). Features are taken before the final layer norm.
* ResNet-18: global-average-pooled feature maps after the stem (layer 0) and after each of the
  8 residual blocks.
* Probe: standardisation + multinomial logistic regression (C = 1, lbfgs), fitted on the first
  n-train training images and scored on the first n-test test images, evaluation transform only.
  The networks themselves saw these training images, so the probe-train accuracy is not reported.
* Depth on the x-axis is relative (layer / number of blocks) so networks of different depth line up.

Writes results/analysis/probes.{json,md,png}.
"""
import argparse
import statistics
import warnings

import numpy as np
import torch

from common import (ALL_RUNS, MODELS, device_and_amp, finished_seeds, load_trained, make_loader, save_json,
                    save_text, style_axes)


@torch.no_grad()
def extract(model, cfg, split, n, device, amp, workers):
    """Returns {"cls": [layer arrays], "mean": [layer arrays]} (ViT) or {"mean": [...]} (ResNet), and labels."""
    loader = make_loader(cfg, split, n=n, batch_size=128, workers=workers)
    is_vit = hasattr(model, "hf")
    feats, labels = None, []
    hooks, captured = [], []
    if not is_vit:
        modules = [model.stem] + list(model.stages)
        for mod in modules:
            hooks.append(mod.register_forward_hook(lambda m, a, o: captured.append(o.float().mean(dim=(2, 3)))))
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                if is_vit:
                    hs = model.hf(pixel_values=x, output_hidden_states=True).hidden_states
                    batch = {"cls": [h[:, 0].float() for h in hs], "mean": [h[:, 1:].float().mean(1) for h in hs]}
                else:
                    captured.clear()
                    model(x)
                    batch = {"mean": list(captured)}
            if feats is None:
                feats = {k: [[] for _ in v] for k, v in batch.items()}
            for k, layers in batch.items():
                for i, f in enumerate(layers):
                    feats[k][i].append(f.cpu().numpy().astype(np.float16))
            labels.append(y.numpy())
    finally:
        for h in hooks:
            h.remove()
    feats = {k: [np.concatenate(parts).astype(np.float32) for parts in layers] for k, layers in feats.items()}
    return feats, np.concatenate(labels)


def probe_accuracy(xtr, ytr, xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(xtr, ytr)
    return float((clf.predict(xte) == yte).mean())


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("cls", "CLS token (ViTs)"), ("mean", "Mean of patch tokens (ViTs), pooled maps (ResNet-18)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (kind, title) in zip(axes, panels):
        for run, r in results.items():
            if kind not in r["layers"]:
                continue
            rows = r["layers"][kind]
            depth = [i / (len(rows) - 1) for i in range(len(rows))]
            mean = [100 * row["mean"] for row in rows]
            std = [100 * row["std"] for row in rows]
            color = MODELS[run][1]
            ax.errorbar(depth, mean, yerr=std, color=color, linewidth=1.6, marker="o", markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.0, capsize=0,
                        label=MODELS[run][0], zorder=3)
        ax.set_title(title, fontsize=10, color="#26251f", loc="left")
        ax.set_xlabel("Relative depth (0 = input embedding, 1 = last block)", fontsize=9, color="#5d5c57")
        ax.set_xlim(-0.03, 1.03)
        style_axes(ax)
    axes[0].set_ylabel("Linear probe test accuracy (%)", fontsize=9, color="#5d5c57")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Where class information appears, layer by layer (mean ± std over seeds)",
                 fontsize=11, color="#26251f", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--runs", nargs="*", default=ALL_RUNS)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    device, amp = device_and_amp()

    results = {}
    for run in args.runs:
        seeds = finished_seeds(run)
        if not seeds:
            print(f"skip {run}: no finished seeds")
            continue
        per_seed = {}
        for seed, seed_dir in seeds:
            model, cfg, _ = load_trained(seed_dir, device, attn="sdpa")
            ftr, ytr = extract(model, cfg, "train", args.n_train, device, amp, args.workers)
            fte, yte = extract(model, cfg, "test", args.n_test, device, amp, args.workers)
            del model
            per_seed[seed] = {k: [probe_accuracy(ftr[k][i], ytr, fte[k][i], yte) for i in range(len(ftr[k]))]
                              for k in ftr}
            print(f"{run} seed {seed}: " + "; ".join(
                f"{k} " + " ".join(f"{100 * a:.1f}" for a in accs) for k, accs in per_seed[seed].items()))
        layers = {}
        for kind in per_seed[next(iter(per_seed))]:
            n_layers = len(per_seed[next(iter(per_seed))][kind])
            rows = []
            for i in range(n_layers):
                vals = [per_seed[s][kind][i] for s in per_seed]
                rows.append({"per_seed": vals, "mean": statistics.mean(vals),
                             "std": statistics.stdev(vals) if len(vals) > 1 else 0.0})
            layers[kind] = rows
        results[run] = {"label": MODELS[run][0], "seeds": sorted(per_seed), "layers": layers}

    if not results:
        print("nothing to analyse")
        return
    save_json("probes.json", {"n_train": args.n_train, "n_test": args.n_test, "models": results})
    lines = ["# Linear probe accuracy by layer\n",
             f"Logistic regression on frozen features; fitted on {args.n_train} training images, scored on "
             f"{args.n_test} test images. Mean ± std over seeds, in %. Layer 0 is the input embedding (ViT) or "
             "stem (ResNet-18). At layer 0 the CLS token is the same for every image, so its probe can only "
             "reach the share of the most common class. Generated by `python src/analysis/probes.py`.\n",
             "| Model | Features | Layer | Test accuracy (%) |", "| --- | --- | --- | --- |"]
    names = {"cls": "CLS token", "mean": "mean of patch tokens / pooled map"}
    for run, r in results.items():
        for kind, rows in r["layers"].items():
            for i, row in enumerate(rows):
                lines.append(f"| {r['label']} | {names[kind]} | {i} | {100 * row['mean']:.2f} ± {100 * row['std']:.2f} |")
    save_text("probes.md", "\n".join(lines) + "\n")
    plot(results, "results/analysis/probes.png")
    print("wrote results/analysis/probes.{json,md,png}")


if __name__ == "__main__":
    main()
