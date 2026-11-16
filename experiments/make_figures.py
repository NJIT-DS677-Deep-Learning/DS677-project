"""Main-results figures for the report (sections B to E), from the finished runs in results/.

    python experiments/make_figures.py

Reads results/<run>/seed*/metrics.json and log.jsonl, plus the FLOP counts in results/model_cost.json
(written by experiments/measure_cost.py, which this script runs first if the file is missing).
Writes results/figures/{claim,compute,curves,per_class}.{png,pdf} and results/figures/figures.md with
every number shown in the figures. Runs on CPU in seconds; nothing is trained or evaluated.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "analysis"))
from common import MODELS, style_axes  # noqa: E402

RESULTS = REPO / "results"
OUT = RESULTS / "figures"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TRAIN_IMAGES = 50_000
MIX_OFF_EPOCH = 175
INK, INK2, MUTED = "#26251f", "#5d5c57", "#8a8984"  # text colors: primary, secondary, muted

# run -> (label, color, hollow marker?, row label in results/model_cost.json)
RUNS = {
    "resnet18": (MODELS["resnet18"][0], MODELS["resnet18"][1], False, "ResNet-18 (CIFAR stem)"),
    "vitlite_p8": (MODELS["vitlite_p8"][0], MODELS["vitlite_p8"][1], False, "ViT-Lite-7/8"),
    "vitlite_p4": (MODELS["vitlite_p4"][0], MODELS["vitlite_p4"][1], False, "ViT-Lite-7/4"),
    "vitlite_p2": (MODELS["vitlite_p2"][0], MODELS["vitlite_p2"][1], False, "ViT-Lite-7/2"),
    "vitlite_convstem": (MODELS["vitlite_convstem"][0], MODELS["vitlite_convstem"][1], False, "ViT-Lite-7 + conv stem"),
    # Same model as vitlite_p2, given the training FLOPs of vitlite_p4 at 300 epochs: same color, hollow marker.
    "vitlite_p2_65ep": ("ViT-Lite-7/2, 65 epochs", MODELS["vitlite_p2"][1], True, "ViT-Lite-7/2"),
}
REFERENCE = "vitb16_finetune"


def load_runs():
    cost_file = RESULTS / "model_cost.json"
    if not cost_file.exists():
        subprocess.run([sys.executable, str(REPO / "experiments/measure_cost.py")], cwd=REPO, check=True)
    gflops = {row["model"]: row["forward_GFLOPs"] for row in json.loads(cost_file.read_text())}
    runs = {}
    for run in list(RUNS) + [REFERENCE]:
        seeds = sorted((RESULTS / run).glob("seed*/metrics.json"))
        if not seeds:
            continue
        ms = [json.loads(p.read_text()) for p in seeds]
        logs = [[json.loads(line) for line in (p.parent / "log.jsonl").read_text().splitlines() if line.strip()]
                for p in seeds]
        acc = np.array([m["final_test_acc"] for m in ms]) * 100
        epochs = ms[0]["epochs"]
        evals = [[(r["epoch"], r["test_acc"] * 100) for r in log if "test_acc" in r] for log in logs]
        common_epochs = sorted(set.intersection(*[{e for e, _ in ev} for ev in evals]))
        curve = np.array([[dict(ev)[e] for e in common_epochs] for ev in evals])
        info = {
            "seeds": len(ms), "acc": acc, "mean": acc.mean(), "std": acc.std(ddof=1) if len(acc) > 1 else 0.0,
            "hours": float(np.mean([m["train_hours"] for m in ms])), "epochs": epochs,
            "per_class": np.mean([m["per_class_acc"] for m in ms], axis=0) * 100,
            "confusion": np.sum([m["confusion_matrix"] for m in ms], axis=0),
            "curve_epochs": np.array(common_epochs), "curve": curve,
        }
        if run in RUNS:
            info["gflops"] = gflops[RUNS[run][3]]
            info["pflop"] = info["gflops"] * TRAIN_IMAGES * epochs / 1e6  # forward FLOPs x training images seen
        runs[run] = info
    return runs


def label(run):
    return RUNS[run][0] if run in RUNS else MODELS[run][0]


def color(run):
    return RUNS[run][1] if run in RUNS else MODELS[run][1]


def point(ax, x, y, run, yerr=None, size=64, zorder=4):
    """Mean marker with a white ring; hollow for the compute-matched run."""
    hollow = RUNS.get(run, (None, None, False))[2]
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color(run), elinewidth=1.4, capsize=3, zorder=zorder - 1)
    ax.scatter(x, y, s=size, color="white" if hollow else color(run), edgecolors=color(run) if hollow else "white",
               linewidths=1.8 if hollow else 1.2, zorder=zorder)


def fig_claim(runs):
    """Accuracy against patch size, with the ResNet-18 baseline and the claim's 1-point threshold."""
    rn = runs["resnet18"]
    order = [("vitlite_p8", 0.0, "8x8 patches\n16 tokens"), ("vitlite_p4", 1.0, "4x4 patches\n64 tokens"),
             ("vitlite_p2", 2.0, "2x2 patches\n256 tokens"), ("vitlite_convstem", 3.35, "Conv stem\n64 tokens")]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    style_axes(ax)
    ax.axhspan(rn["mean"] - rn["std"], rn["mean"] + rn["std"], color=color("resnet18"), alpha=0.14, lw=0, zorder=1)
    ax.axhline(rn["mean"], color=color("resnet18"), lw=1.8, zorder=2)
    ax.axhline(rn["mean"] - 1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    if REFERENCE in runs:
        ax.axhline(runs[REFERENCE]["mean"], color=color(REFERENCE), lw=1.4, ls=(0, (1, 2)), zorder=2)
        ax.text(-0.42, runs[REFERENCE]["mean"] + 0.18, f"ViT-B/16, pretrained + fine-tuned: {runs[REFERENCE]['mean']:.2f}%"
                " (reference, 224 px)", fontsize=8.5, color=INK2, va="bottom")
    ax.text(-0.42, rn["mean"] + 0.18, f"ResNet-18 baseline: {rn['mean']:.2f} ± {rn['std']:.2f}%", fontsize=8.5,
            color=INK, va="bottom")
    ax.text(-0.42, rn["mean"] - 1.0 - 0.18, f"Claim threshold: 1 point below ResNet-18 ({rn['mean'] - 1:.2f}%)",
            fontsize=8.5, color=INK2, va="top")
    means = []
    rng = np.random.default_rng(0)
    for run, x, _ in order:
        r = runs[run]
        jitter = rng.uniform(-0.07, 0.07, len(r["acc"]))
        ax.scatter(x + 0.2 + jitter, r["acc"], s=18, color=color(run), alpha=0.55, lw=0, zorder=3)
        point(ax, x, r["mean"], run, yerr=r["std"], size=70)
        ax.text(x, r["mean"] - max(r["std"], 0.25) - 0.3, f"{r['mean']:.2f}%\n{rn['mean'] - r['mean']:.2f} below",
                fontsize=8.5, color=INK, ha="center", va="top", linespacing=1.15)
        means.append((x, r["mean"]))
    xs, ys = zip(*means[:3])
    ax.plot(xs, ys, color=MUTED, lw=1.2, zorder=2)
    ax.axvline(2.68, color="#d6d5cf", lw=1.0, zorder=1)
    ax.text(3.35, 86.35, "extension", fontsize=8, color=MUTED, ha="center")
    ax.set_xticks([x for _, x, _ in order], [t for _, _, t in order])
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylim(86, 100)
    ax.set_ylabel("Test accuracy (%)", fontsize=9.5, color=INK2)
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=INK)
    handles = [Line2D([], [], marker="o", ls="", markersize=7.5, markerfacecolor=color(r), markeredgecolor="white",
                      label=label(r)) for r, _, _ in order]
    handles += [Line2D([], [], marker="o", ls="", markersize=4, markerfacecolor=MUTED, markeredgecolor="none",
                       alpha=0.7, label="single seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, handletextpad=0.3,
               columnspacing=1.2)
    fig.suptitle("No patch size brings ViT-Lite-7 within 1 point of ResNet-18 (mean ± std over 3 seeds)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "claim")


def fig_compute(runs):
    """Accuracy against training compute and against measured GPU-hours."""
    order = ["vitlite_p8", "vitlite_p4", "vitlite_convstem", "vitlite_p2", "vitlite_p2_65ep", "resnet18"]
    short = {"resnet18": "ResNet-18", "vitlite_p8": "7/8", "vitlite_p4": "7/4", "vitlite_p2": "7/2",
             "vitlite_convstem": "conv stem", "vitlite_p2_65ep": "7/2, 65 epochs"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    for ax, key, xlabel in [(axes[0], "pflop", "Training compute (PFLOP, forward passes, log scale)"),
                            (axes[1], "hours", "GPU-hours per run (A100 MIG 2g.10gb slice)")]:
        style_axes(ax)
        ax.grid(axis="x", color="#efeee9", linewidth=0.6)
        p2, p2s = runs["vitlite_p2"], runs["vitlite_p2_65ep"]
        ax.plot([p2s[key], p2[key]], [p2s["mean"], p2["mean"]], color=MUTED, lw=1.0, ls=(0, (2, 2)), zorder=2)
        for run in order:
            r = runs[run]
            point(ax, r[key], r["mean"], run, yerr=r["std"])
            dx, ha = (1.12, "left") if key == "pflop" else (0.08, "left")
            x_text = r[key] * dx if key == "pflop" else r[key] + dx
            y_text = r["mean"] - 1.0 if run == "vitlite_p2_65ep" else r["mean"]
            if run == "vitlite_p2" and key == "pflop":
                x_text, ha = r[key] / 1.12, "right"
            ax.text(x_text, y_text, short[run], fontsize=8.5, color=INK, ha=ha, va="center")
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1.2, 60)
    axes[0].set_xticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    axes[0].minorticks_off()
    p4 = runs["vitlite_p4"]
    axes[0].axvline(p4["pflop"], color="#d6d5cf", lw=1.0, zorder=1)
    axes[0].text(p4["pflop"] / 1.06, 83.5, "same training\nFLOPs", fontsize=8, color=MUTED, va="center", ha="right")
    axes[1].set_xlim(0, 3.9)
    axes[0].set_ylim(76, 98)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=9.5, color=INK2)
    handles = [Line2D([], [], marker="o", ls="", markersize=7.5, markerfacecolor="white" if RUNS[r][2] else color(r),
                      markeredgecolor=color(r) if RUNS[r][2] else "white", markeredgewidth=1.6 if RUNS[r][2] else 1,
                      label=label(r)) for r in ["resnet18"] + order[:-1]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, handletextpad=0.3)
    fig.suptitle("At equal training FLOPs, 2x2 patches fall far behind 4x4 patches",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "compute")


def fig_curves(runs):
    """Test accuracy during training, by epoch and by cumulative training compute."""
    order = ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem", "vitlite_p2_65ep"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    for ax, by_compute in [(axes[0], False), (axes[1], True)]:
        style_axes(ax)
        for run in order:
            r = runs[run]
            x = r["curve_epochs"].astype(float)
            if by_compute:
                x = x * r["gflops"] * TRAIN_IMAGES / 1e6
            ls = (0, (3, 2)) if RUNS[run][2] else "-"
            ax.fill_between(x, r["curve"].min(0), r["curve"].max(0), color=color(run), alpha=0.15, lw=0)
            ax.plot(x, r["curve"].mean(0), color=color(run), lw=1.8, ls=ls)
    axes[0].axvline(MIX_OFF_EPOCH, color="#d6d5cf", lw=1.0, zorder=1)
    axes[0].text(MIX_OFF_EPOCH - 4, 42, "Mixup and CutMix\noff from here", fontsize=8, color=MUTED, ha="right")
    axes[0].set_xlim(0, 305)
    axes[0].set_xlabel("Epoch", fontsize=9, color=INK2)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    axes[1].minorticks_off()
    axes[1].set_xlabel("Training compute so far (PFLOP, log scale)", fontsize=9, color=INK2)
    axes[0].set_ylim(35, 100)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=9.5, color=INK2)
    handles = [Line2D([], [], color=color(r), lw=1.8, ls=(0, (3, 2)) if RUNS[r][2] else "-", label=label(r))
               for r in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Test accuracy during training (line: mean over seeds; band: seed range)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "curves")


def fig_per_class(runs):
    """Per-class test accuracy, classes sorted by the ResNet-18 minus ViT-Lite-7/4 gap."""
    order = ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem"]
    gap = runs["resnet18"]["per_class"] - runs["vitlite_p4"]["per_class"]
    rows = list(np.argsort(gap))  # smallest gap at the bottom, largest at the top
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    for i, c in enumerate(rows):
        vals = [runs[r]["per_class"][c] for r in order]
        ax.plot([min(vals), max(vals)], [i, i], color="#d6d5cf", lw=1.0, zorder=1)
        for r, v, dy in zip(order, vals, np.linspace(0.28, -0.28, len(order))):
            ax.scatter(v, i + dy, s=46, color=color(r), edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(101.2, i, f"{gap[c]:.1f}", fontsize=8.5, color=INK, va="center", ha="right")
    ax.text(101.2, len(rows) - 0.35, "ResNet-18\nminus 7/4", fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_yticks(range(len(rows)), [CLASSES[c] for c in rows], fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(70, 101.5)
    ax.set_ylim(-0.6, len(rows) + 0.5)
    ax.set_xlabel("Per-class test accuracy (%), mean over 3 seeds", fontsize=9, color=INK2)
    handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=color(r), markeredgecolor="white",
                      label=label(r)) for r in order]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, handletextpad=0.3,
               columnspacing=1.0)
    fig.suptitle("The ViTs lose most on animals (cat, dog, bird, deer) and least on vehicles and frogs",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "per_class")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def write_tables(runs):
    rn = runs["resnet18"]
    lines = ["# Main-results figures: the numbers behind them", "",
             "Generated by `python experiments/make_figures.py` from `results/<run>/seed*/`. Test accuracy is the "
             "final-epoch value; training compute is forward GFLOPs per image x 50,000 images x epochs (the "
             "backward pass adds roughly twice as much again for every model).", "",
             "## Accuracy and cost (`claim.png`, `compute.png`)", "",
             "| Run | Seeds | Test accuracy (%) | ResNet-18 minus this run (points) | Forward GFLOPs/image | Epochs | "
             "Training compute (PFLOP) | GPU-hours per run |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem", "vitlite_p2_65ep",
                REFERENCE]:
        if run not in runs:
            continue
        r = runs[run]
        acc = f"{r['mean']:.2f} ± {r['std']:.2f}" if r["seeds"] > 1 else f"{r['mean']:.2f} (1 seed)"
        gap = "" if run == "resnet18" else f"{rn['mean'] - r['mean']:.2f}"
        g = f"{r['gflops']:.3f}" if "gflops" in r else "35.126 (224 px)"
        pf = f"{r['pflop']:.1f}" if "pflop" in r else "n/a (pretrained)"
        lines.append(f"| {label(run)} | {r['seeds']} | {acc} | {gap} | {g} | {r['epochs']} | {pf} | {r['hours']:.2f} |")
    lines += ["", "## Per-class test accuracy (`per_class.png`), mean over seeds, %", "",
              "| Class | " + " | ".join(label(r) for r in RUNS if r != "vitlite_p2_65ep") + " | ResNet-18 minus 7/4 |",
              "| --- " * (len(RUNS) + 1) + "|"]
    for c, name in enumerate(CLASSES):
        vals = " | ".join(f"{runs[r]['per_class'][c]:.2f}" for r in RUNS if r != "vitlite_p2_65ep")
        lines.append(f"| {name} | {vals} | {runs['resnet18']['per_class'][c] - runs['vitlite_p4']['per_class'][c]:.2f} |")
    lines += ["", "## Most frequent confusions (true class -> predicted class, test images summed over seeds)", "",
              "| Run | 1st | 2nd | 3rd | 4th |", "| --- | --- | --- | --- | --- |"]
    for run in ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem", REFERENCE]:
        off = runs[run]["confusion"].copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off.ravel(), kind="stable")[:4], off.shape))[0]
        cells = " | ".join(f"{CLASSES[i]} -> {CLASSES[j]} ({int(off[i, j])})" for i, j in top)
        lines.append(f"| {label(run)} ({runs[run]['seeds']} seed{'s' if runs[run]['seeds'] > 1 else ''}) | {cells} |")
    lines += ["", "## Test accuracy during training (`curves.png`), mean over seeds, %", ""]
    header_epochs = [10, 30, 50, 100, 150, 170, 180, 200, 250, 300]
    lines += ["| Run | " + " | ".join(f"epoch {e}" for e in header_epochs) + " |",
              "| --- " * (len(header_epochs) + 1) + "|"]
    for run in ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem"]:
        r = runs[run]
        m = dict(zip(r["curve_epochs"].tolist(), r["curve"].mean(0)))
        lines.append(f"| {label(run)} | " + " | ".join(f"{m[e]:.2f}" if e in m else "" for e in header_epochs) + " |")
    r = runs["vitlite_p2_65ep"]
    lines.append("| ViT-Lite-7/2, 65 epochs | " + ", ".join(f"epoch {e}: {v:.2f}" for e, v in
                                                           zip(r["curve_epochs"], r["curve"].mean(0))) + " |")
    (OUT / "figures.md").write_text("\n".join(lines) + "\n")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10})
    runs = load_runs()
    fig_claim(runs)
    fig_compute(runs)
    fig_curves(runs)
    fig_per_class(runs)
    write_tables(runs)
    print(f"wrote {OUT}/{{claim,compute,curves,per_class}}.{{png,pdf}} and figures.md")


if __name__ == "__main__":
    main()
