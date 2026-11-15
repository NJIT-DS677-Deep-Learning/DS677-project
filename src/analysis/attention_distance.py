"""Mean attention distance per head and layer (research question 3).

    python src/analysis/attention_distance.py --n 1000

For every attention head, the average distance between a patch and the patches it attends to,
weighted by the attention weights: the measure in Figure 7 of Dosovitskiy et al. (2021) and
Figure 4 of Raghu et al. (2021). A CNN's early layers are local by construction; this asks whether
a ViT trained from scratch on CIFAR-10 learns local heads in its early layers at all.

Details, fixed before looking at results:
* Only patch tokens count, as queries and as keys. The CLS token is dropped and each query's
  attention over patch keys is renormalised to sum to 1.
* Distances run between patch centres and are divided by the image width, so 32 px models and
  ViT-B/16 at 224 px are comparable; they are reported in original CIFAR pixels (x 32).
* Averaged over the first n test images and all query patches. Each seed is a separate model;
  heads are not matched across seeds.
* The reference line is the distance under uniform attention over all patches.

Writes results/analysis/attention_distance.{json,md,png}.
"""
import argparse

import torch

from common import (CIFAR_PX, MODELS, PRETRAINED_ONLY, VIT_RUNS, device_and_amp, finished_seeds,
                    image_size, load_pretrained_only, load_trained, make_loader, save_json, save_text,
                    style_axes, token_grid_distances)


@torch.no_grad()
def head_distances(model, cfg, n, device, amp, workers):
    loader = make_loader(cfg, "test", n=n, batch_size=128, workers=workers)
    img = image_size(cfg)
    sums, count, dist = None, 0, None
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out = model.hf(pixel_values=x, output_attentions=True)
        attns = out.attentions
        if not attns:
            raise RuntimeError("model returned no attention maps; it must be built with attn='eager'")
        if dist is None:
            dist = token_grid_distances(attns[0].shape[-1] - 1, img).to(device)
            sums = torch.zeros(len(attns), attns[0].shape[1], device=device, dtype=torch.float64)
        for i, a in enumerate(attns):
            a = a[:, :, 1:, 1:].float()                      # patch queries x patch keys
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)  # renormalise without CLS
            d = (a * dist).sum(-1)                            # (B, heads, queries)
            sums[i] += d.mean(-1).sum(0).double()
        count += x.shape[0]
    uniform = dist.mean().item()
    return (sums / count).tolist(), uniform


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = list(results)
    cols = min(3, len(runs))
    rows = (len(runs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.3 * rows), sharey=True, squeeze=False)
    ymax = 0
    for k, run in enumerate(runs):
        ax = axes[k // cols][k % cols]
        r = results[run]
        color = MODELS[run][1]
        n_seeds = len(r["seeds"])
        for s_i, layers in enumerate(r["seeds"].values()):
            for li, heads in enumerate(layers):
                offs = (s_i - (n_seeds - 1) / 2) * 0.12
                ys = [h * CIFAR_PX for h in heads]
                ymax = max(ymax, max(ys))
                ax.scatter([li + 1 + offs] * len(ys), ys, s=26, color=color, alpha=0.85,
                           edgecolors="white", linewidths=0.6, zorder=3)
        u = r["uniform_reference"] * CIFAR_PX
        ax.axhline(u, color="#8a8984", linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
        ax.text(r["n_layers"] + 0.45, u + 0.3, "uniform\nattention", va="bottom", ha="left", fontsize=8, color="#5d5c57")
        ax.set_title(MODELS[run][0], fontsize=10, color="#26251f", loc="left")
        ax.set_xticks(range(1, r["n_layers"] + 1))
        ax.set_xlim(0.4, r["n_layers"] + 2.2)
        ax.set_xlabel("Layer", fontsize=9, color="#5d5c57")
        style_axes(ax)
    for k in range(len(runs), rows * cols):
        axes[k // cols][k % cols].axis("off")
    for row in axes:
        row[0].set_ylabel("Mean attention distance\n(CIFAR pixels)", fontsize=9, color="#5d5c57")
    axes[0][0].set_ylim(0, max(ymax, 1) * 1.1)
    fig.suptitle("How far each attention head looks, by layer (one dot per head per seed)",
                 fontsize=11, color="#26251f", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000, help="test images to average over")
    p.add_argument("--runs", nargs="*", default=VIT_RUNS)
    p.add_argument("--no-pretrained-only", action="store_true", help="skip the ImageNet-21k checkpoint before fine-tuning")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    device, amp = device_and_amp()

    results = {}
    for run in args.runs:
        seeds = finished_seeds(run)
        if not seeds:
            print(f"skip {run}: no finished seeds")
            continue
        entry = {"label": MODELS[run][0], "seeds": {}}
        for seed, seed_dir in seeds:
            model, cfg, _ = load_trained(seed_dir, device)
            layers, uniform = head_distances(model, cfg, args.n, device, amp, args.workers)
            entry["seeds"][str(seed)] = layers
            entry.update(n_layers=len(layers), uniform_reference=uniform, image_size=image_size(cfg))
            print(f"{run} seed {seed}: layer means (CIFAR px) "
                  + ", ".join(f"{CIFAR_PX * sum(h) / len(h):.1f}" for h in layers))
            del model
        results[run] = entry

    if not args.no_pretrained_only:
        try:
            model, cfg = load_pretrained_only(device)
            layers, uniform = head_distances(model, cfg, args.n, device, amp, args.workers)
            results[PRETRAINED_ONLY] = {"label": MODELS[PRETRAINED_ONLY][0], "seeds": {"pretrained": layers},
                                        "n_layers": len(layers), "uniform_reference": uniform,
                                        "image_size": image_size(cfg)}
            print(f"{PRETRAINED_ONLY}: layer means (CIFAR px) "
                  + ", ".join(f"{CIFAR_PX * sum(h) / len(h):.1f}" for h in layers))
        except Exception as e:  # checkpoint not cached, or offline
            print(f"skip {PRETRAINED_ONLY}: {type(e).__name__}: {e}")

    if not results:
        print("nothing to analyse")
        return
    save_json("attention_distance.json", {"n_images": args.n, "units": "fraction of image width", "models": results})

    lines = ["# Mean attention distance by layer\n",
             f"Averaged over the first {args.n} test images, patch tokens only, in original CIFAR pixels "
             "(32 = full image width). Mean, lowest and highest head across all seeds. "
             "Generated by `python src/analysis/attention_distance.py`.\n",
             "| Model | Layer | Mean (px) | Lowest head (px) | Highest head (px) | Uniform attention (px) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for run, r in results.items():
        for li in range(r["n_layers"]):
            heads = [h * CIFAR_PX for layers in r["seeds"].values() for h in layers[li]]
            lines.append(f"| {r['label']} | {li + 1} | {sum(heads) / len(heads):.1f} | {min(heads):.1f} | "
                         f"{max(heads):.1f} | {r['uniform_reference'] * CIFAR_PX:.1f} |")
    save_text("attention_distance.md", "\n".join(lines) + "\n")
    plot(results, "results/analysis/attention_distance.png")
    print("wrote results/analysis/attention_distance.{json,md,png}")


if __name__ == "__main__":
    main()
