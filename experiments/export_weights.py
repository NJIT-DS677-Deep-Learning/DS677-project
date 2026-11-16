"""Export the trained weights the demo notebook uses (seed 0 of every from-scratch model).

    python experiments/export_weights.py        # on Wulver, where results/<run>/seed0/last.pt exist

Writes demo/weights/<run>.safetensors and demo/weights/manifest.json. Weights are stored in float16
(about 7.5 MB per ViT-Lite-7 and 22 MB for ResNet-18) so they fit in the repository; the notebook
loads them back as float32 and reports the test accuracy it measures next to the training run's.
Training checkpoints (last.pt, with optimizer state) stay out of git. ViT-B/16 (172 MB even in
float16) is not exported; the notebook evaluates it only where its checkpoint exists, i.e. on Wulver.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
RUNS = ["resnet18", "vitlite_p8", "vitlite_p4", "vitlite_p2", "vitlite_convstem"]
SEED = 0
OUT = REPO / "demo" / "weights"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "dtype": "float16", "transformers_version": version("transformers"),
                "torch_version": torch.__version__, "runs": {}}
    for run in RUNS:
        seed_dir = REPO / "results" / run / f"seed{SEED}"
        ck = torch.load(seed_dir / "last.pt", map_location="cpu", weights_only=False)
        state = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in ck["model"].items()}
        path = OUT / f"{run}.safetensors"
        save_file(state, str(path), metadata={"run": run, "seed": str(SEED)})
        cfg = yaml.safe_load((seed_dir / "config.yaml").read_text())
        metrics = json.loads((seed_dir / "metrics.json").read_text())
        manifest["runs"][run] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "params": sum(v.numel() for k, v in ck["model"].items() if v.is_floating_point()
                          and "running_" not in k),
            "final_test_acc": metrics["final_test_acc"],
            "train_hours": metrics["train_hours"],
            "config": {"model": cfg["model"], "data": cfg["data"]},
        }
        print(f"{run}: {path.stat().st_size / 1e6:.1f} MB, run's final test accuracy {metrics['final_test_acc']:.4f}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT}/manifest.json")


if __name__ == "__main__":
    main()
