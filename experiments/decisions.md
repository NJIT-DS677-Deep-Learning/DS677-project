# Decision log

Choices made during the project, recorded when they were made, so the report can state what was decided before which results were seen.

## 2026-11-13: learning rates

**30-epoch sweep** (`experiments/wulver/lr_sweep.sh`; seed 0; 45,000 train / 5,000 validation images; test set unused):

| Model | Learning rate | Validation accuracy (%) |
| --- | --- | --- |
| ResNet-18, SGD | 0.05 | 93.38 |
| ResNet-18, SGD | 0.1 | 92.86 |
| ResNet-18, SGD | 0.2 | 92.62 |
| ViT-Lite-7/4, AdamW | 2.75e-4 | 75.56 |
| ViT-Lite-7/4, AdamW | 5.5e-4 (published CCT recipe) | 67.84 |
| ViT-Lite-7/4, AdamW | 1.1e-3 | 64.18 |

**Observation.** For both models the best rate was the lowest value tried, so the grid did not bracket the optimum. Short training runs are known to favor smaller learning rates than the full schedule would (short-horizon bias, Wu et al., 2018), and the published ViT-Lite recipe uses 5.5e-4 for 300 epochs. After 30 epochs ViT-Lite-7/4 reached 75.6% against ResNet-18's 93.4%, a first sign of how much slower the ViT trains from scratch.

**Decision, made before any test-set run.** Repeat the comparison at the full 300-epoch schedule on the same validation split, with each grid extended one step lower: ResNet-18 at 0.025, 0.05 and 0.1; ViT-Lite-7/4 at 1.375e-4, 2.75e-4 and 5.5e-4 (`experiments/wulver/lr_check_full.sh`). The highest full-length validation accuracy decides each model's rate, and the ViT rate applies to every ViT-Lite-7 config. If the best is again at an edge of the grid, it is used as is and reported, without extending the grid further. The ViT-B/16 fine-tune keeps its fixed recipe and is not swept.

**Full-length check** (`experiments/wulver/lr_check_full.sh`; 300 epochs; seed 0; same validation split; test set unused):

| Model | Learning rate | Validation accuracy (%) |
| --- | --- | --- |
| ResNet-18, SGD | 0.025 | 96.08 |
| ResNet-18, SGD | 0.05 | 95.82 |
| ResNet-18, SGD | 0.1 | 95.80 |
| ViT-Lite-7/4, AdamW | 1.375e-4 | 90.48 |
| ViT-Lite-7/4, AdamW | 2.75e-4 | 91.22 |
| ViT-Lite-7/4, AdamW | 5.5e-4 (published CCT recipe) | 90.20 |

**Result.** ViT-Lite-7/4: the best rate, 2.75e-4, is now bracketed by worse values on both sides. The published 5.5e-4 trailed it by 7.7 points after 30 epochs but by only 1.0 point after 300, so the short sweep exaggerated the difference. ResNet-18: 0.025 is again the lowest value tried, but the three rates lie within 0.3 points of each other on a single seed.

**Final rates, applied to the configs before any main run.** Following the rule above: ResNet-18 uses 0.025, and every ViT-Lite-7 config (patch 8, 4, 2 and the conv stem) uses 2.75e-4. Because the ViTs no longer use the published learning rate, a gap to the published ViT-Lite numbers in the replication anchor may partly come from this choice; the report will say so.

## 2026-11-14: ViT-B/16 replication anchor

The fine-tuned `google/vit-base-patch16-224-in21k` (224 px, 5 epochs, seed 0) reached 98.77% test accuracy, against 98.95% reported by Dosovitskiy et al. (2021) for ViT-B/16 fine-tuned at 384 px: 0.18 points lower, at about 0.44 GPU-hours.

## 2026-11-15: extensions (conv stem and compute-matched 2x2)

**Setup, fixed before the runs** (`experiments/wulver/submit_extensions.sh`). Both runs use the shared recipe and the 2.75e-4 learning rate.

- **Conv stem.** Two stages of (3x3 convolution, ReLU, 3x3 max-pool with stride 2) replace the patch embedding. They produce the same 8x8 grid of 64 tokens as 4x4 patches, so only the tokenizer changes. Each token sees a 13x13-pixel window, and neighbouring tokens overlap.
- **Compute-matched 2x2.** ViT-Lite-7/2 gets the training FLOPs of ViT-Lite-7/4 at 300 epochs. 2x2 patches cost 2.361 / 0.509 = 4.64 times the forward FLOPs per image (`experiments/measure_cost.py`), and 300 / 4.64 = 64.7, rounded to 65 epochs. The schedule is scaled with it: 2 warmup and 2 cooldown epochs, and Mixup and CutMix off from epoch 38 (the same 175/300 of the way as in the full runs).

**Results** (3 seeds each, final-epoch test accuracy, `results/summary.md`):

| Run | Test accuracy (%) | Forward GFLOPs per image | Train hours per run |
| --- | --- | --- | --- |
| ViT-Lite-7/4, 300 epochs | 91.57 ± 0.49 | 0.509 | 1.04 |
| ViT-Lite-7 + conv stem, 300 epochs | 93.08 ± 0.22 | 0.586 | 1.00 |
| ViT-Lite-7/2, 300 epochs | 91.71 ± 0.35 | 2.361 | 3.42 |
| ViT-Lite-7/2, 65 epochs | 78.53 ± 1.02 | 2.361 | 0.67 |

**Limits of the compute-matched run, recorded after seeing its result.**

1. It reuses the 2.75e-4 rate chosen at 300 epochs. The 30-epoch sweep showed that short schedules hurt ViT-Lite badly (75.6% validation accuracy after 30 epochs, 91.2% after 300), so part of its 13.0-point gap to ViT-Lite-7/4 is a short-schedule penalty and not an effect of patch size alone.
2. The match is in FLOPs, not in time. On the MIG slice the run took 0.67 hours against 1.04 for ViT-Lite-7/4, because the larger matrices of 2x2 patches use the GPU more efficiently. A time-matched run would have about 91 epochs (300 × 1.04 / 3.42).

**Decision.** The report describes this result as "at equal training FLOPs and an unchanged learning rate, 2x2 patches fall far behind 4x4 patches", states both limits, and does not present the gap as the cost of patch size alone.
