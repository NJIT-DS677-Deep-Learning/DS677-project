# Report outline (Milestone 3)

Working title: **Why a Small Vision Transformer Trails a ResNet-18 on CIFAR-10. A Patch-Size Study with Layer Analyses**

Rishabh Verma. DS 677 Deep Learning, Fall 2026, Model Review and Demo track.

This outline maps every section of the report to the evidence already in the repository. Every number below comes from a file in `results/` or `experiments/decisions.md`, and the file is named next to it. Sections follow the Project Guide skeleton and the Milestone 3 rubric (4 points each).

## 0. Front matter

- **Page 1 needs** (required by Milestone 3):
  - the title, the author and the course;
  - the repository link, https://github.com/NJIT-DS677-Deep-Learning/DS677-project;
  - the video link (to add once the video is recorded).
- **Abstract** (about 150 words): claim, setup, main result, mechanism, outlook.
  - Main result: 8.15, 4.54 and 4.39 points below ResNet-18 at patch sizes 8, 4 and 2.
  - Mechanism: no locality at the input; the conv stem helps (93.08%) but the gap moves to the deeper layers.
- **Format:** LaTeX (`article`, 11 pt) with the source in `report/` and the PDF built from it. The figures already exist as vector PDFs in `results/figures/`.

## 1. Model and Claim (rubric A)

Rubric: accuracy and clarity of the architecture description, quality of the comparison with related work, clear restatement of the claim.

**1.1 The Vision Transformer**
- Patchify, then a linear embedding, a CLS token, learned position embeddings and a pre-norm encoder (multi-head self-attention and an MLP, each with a residual connection). The classification head reads the final CLS token.
- Write the four ViT equations (Dosovitskiy et al., 2021): the embedding z0, the two residual updates and the output y.
- Contrast with a CNN: no locality and no weight sharing across positions, only what the position embeddings and the data teach. This is the "weak inductive bias" every later section tests.

**1.2 The models studied** (Table 1, from `results/model_cost.json`):

| Model | Tokens (with CLS) | Params (M) | Forward GFLOPs/image |
| --- | --- | --- | --- |
| ResNet-18 (CIFAR stem) | n/a | 11.17 | 1.111 |
| ViT-Lite-7/8 | 17 | 3.75 | 0.128 |
| ViT-Lite-7/4 | 65 | 3.72 | 0.509 |
| ViT-Lite-7/2 | 257 | 3.76 | 2.361 |
| ViT-Lite-7 + conv stem | 65 | 3.86 | 0.586 |
| ViT-B/16, 224 px | 197 | 85.81 | 35.126 |

- **ViT-Lite-7** (Hassani et al., 2021): 7 layers, 4 heads, width 256, MLP 512. It is built with Hugging Face `ViTForImageClassification` (`src/models.py`), and our parameter counts match the paper's 3.72M for 7/4.
- **The conv stem:** two (3x3 conv, ReLU, 3x3 max-pool with stride 2) stages giving the same 64 tokens as 4x4 patches. Each token sees an overlapping 13x13 window.
- **ViT-B/16:** pretrained on ImageNet-21k, the reference for what pretraining buys.
- **Figure 1 (to draw):** a pipeline diagram, image to patches to tokens to 7 blocks to the CLS token to logits, with the conv stem as an alternative tokenizer. It can reuse the shapes printed in the demo notebook, section 2.
- **What the patch embedding learns** (optional Figure 9, from the demo notebook, section 2, step 3): ViT-Lite-7/4's position embeddings, with nothing telling the model the patches form a grid, come out most similar to their neighbours on the 8x8 grid. This reproduces the pattern Dosovitskiy et al. (2021) show for ViT, and is a good picture for the video.

**1.3 Related work**, grouped by how each paper attacks the ViT's data hunger (all checked on arXiv):
- **Scale and recipes.**
  - ViT needs large pretraining datasets to beat CNNs (Dosovitskiy et al., 2021).
  - DeiT trains on ImageNet-1k with strong augmentation and distillation (Touvron et al., 2021).
  - AugReg finds the right augmentation and regularization worth about a 10x larger dataset (Steiner et al., 2022).
- **Compact transformers for small data** (Hassani et al., 2021): ViT-Lite, then CVT (SeqPool replaces the CLS token) and CCT (a convolutional tokenizer).
  - On CIFAR-10 at 300 epochs they report 89.10% (ViT-Lite-7/8), 93.57% (7/4), 94.01% (CVT-7/4), 95.04% (CCT-7/3x2) and 96.53% (CCT-7/3x1).
- **Putting locality back:**
  - convolutional stems improve optimizability and accuracy (Xiao et al., 2021);
  - soft convolutional bias through gated positional self-attention (ConViT, d'Ascoli et al., 2021);
  - shifted patch tokenization and locality self-attention for small datasets (Lee, Lee and Song, 2021).
- **Patch size as a compute lever:**
  - FlexiViT shows smaller patches trade compute for accuracy (Beyer et al., 2023);
  - single-pixel tokens (Nguyen et al., 2024).
- **Analysis tools:** mean attention distance and layer-wise representations, the methods we reuse (Raghu et al., 2021).
- **Where this project fits:** it measures the patch-size trade-off at 32 px from scratch against a tuned CNN, and uses layer analyses to locate where the ViT falls behind.

**1.4 Claim and research questions**
- Restate the approved claim verbatim (README).
- **RQ1:** can any patch size reach ResNet-18?
- **RQ2:** accuracy against compute.
- **RQ3:** what the ViT fails to learn, layer by layer.
- **RQ4:** does a conv stem beat smaller patches?

## 2. Baseline (rubric B)

Rubric: the baseline is sensible, given a fair chance, and clearly reported.

- **Why ResNet-18:** it is the standard CIFAR CNN (He et al., 2016), in its CIFAR form: a 3x3 stem with stride 1 and no max-pool.
- **Fair chance:**
  - It gets the same data, augmentation (RandAugment m9, Mixup 0.8 and CutMix 1.0 switched off at epoch 175, label smoothing 0.1), 300 epochs and batch size 128.
  - It uses its own customary optimizer: SGD with Nesterov momentum and weight decay 5e-4.
  - Its learning rate was tuned with the same two-stage protocol as the ViT, on a 5,000-image validation split (`experiments/decisions.md`).
- **Table 2, learning-rate selection** (both models, validation accuracy):

| Model | 30 epochs | 300 epochs |
| --- | --- | --- |
| ResNet-18 | 93.38 / 92.86 / 92.62 at 0.05 / 0.1 / 0.2 | 96.08 / 95.82 / 95.80 at 0.025 / 0.05 / 0.1 |
| ViT-Lite-7/4 | 75.56 / 67.84 / 64.18 at 2.75e-4 / 5.5e-4 / 1.1e-3 | 90.48 / 91.22 / 90.20 at 1.375e-4 / 2.75e-4 / 5.5e-4 |

  - The 30-epoch sweep favoured the smallest rate for both models. This is short-horizon bias (Wu et al., 2018), which is why the 300-epoch check decided.
- **Result:** 96.10 ± 0.13% over 3 seeds (96.25, 95.99, 96.07) at 1.09 GPU-hours per run (`results/summary.md`).
  - Best-epoch reporting would add only 0.03 points.
  - Its learning curve leads every ViT at every evaluation; at epoch 10 it is at 81.74%, against 58.61% for ViT-Lite-7/4 (`results/figures/figures.md`, Figure 4).
- **Fairness in both directions** (stated openly):
  - ResNet-18 has 3x the parameters of ViT-Lite-7.
  - ViT-Lite-7/2 uses 2.1x ResNet-18's FLOPs per image and is still 4.39 points behind.

## 3. Replication Anchor (rubric C)

Rubric: fidelity of the replication, honest handling of discrepancies with published values.

**Table 3** (3 seeds unless noted; final-epoch test accuracy; `results/summary.md`):

| Model | Ours (%) | Published (%) | Difference (points) |
| --- | --- | --- | --- |
| ViT-Lite-7/8 | 87.96 ± 0.29 | 89.10 (Hassani et al., 2021) | -1.14 |
| ViT-Lite-7/4 | 91.57 ± 0.49 | 93.57 (Hassani et al., 2021) | -2.00 |
| ViT-B/16, fine-tuned (1 seed) | 98.77 | 98.95 (Dosovitskiy et al., 2021, at 384 px) | -0.18 |

- **ViT-B/16 is faithful.** It lands 0.18 below the published value with a cheaper setup: 224 px instead of 384, and 5 epochs.
- **ViT-Lite lands 1 to 2 points low.** The investigation, in order:
  1. **Reporting.** We report the final epoch. Reporting the best epoch would add only 0.12 (7/8) and 0.35 (7/4) points (`log.jsonl`), so the reporting choice explains little.
  2. **Learning rate.** We use 2.75e-4 instead of the published 5.5e-4. At 300 epochs on the validation split, 5.5e-4 scored 1.02 points lower (single seed), so our choice should not cost accuracy.
  3. **Seeds.** Even our best ViT-Lite-7/4 seed (91.97%) is 1.60 below 93.57, so this is not seed noise.
  4. **Regularization inside the model (likely the main cause, not tested).** The released ViT-Lite code (`src/vit.py` in SHI-Labs/Compact-Transformers) uses attention dropout 0.1 and stochastic depth 0.1 by default. Our Hugging Face ViT uses neither. AugReg (Steiner et al., 2022) shows how much this kind of regularization matters when data is small.
  5. **Augmentation implementation.** Their config uses timm's `rand-m9-mstd0.5-inc1` RandAugment and random interpolation; we use torchvision's RandAugment(2, 9). It is the same idea, not the same operations.
- **Conv stem as a third check.**
  - CCT-7/3x2 uses a two-layer convolutional tokenizer and SeqPool at the same 0.29 GMACs as our conv stem, and reports 95.04%. Ours reaches 93.08 ± 0.22%, which is 1.96 lower.
  - The gap matches the ViT-Lite-7/4 gap (2.00), which points to a recipe difference shared by all our ViTs rather than a bug in one model.
  - It is not an exact replication: CCT uses SeqPool, we use a CLS token.
- **The demo checks the weights.** It reloads the exported weights and reproduces each seed-0 run's logged test accuracy (`demo/demo.ipynb`, section 5).

## 4. Perturbation Study (rubric D)

Rubric: soundness of the experimental setup, quality of results and figures.

- **Setup.** This is a sensitivity sweep over one variable, the tokenizer: 8x8, 4x4 and 2x2 patches.
  - Everything else is held fixed: architecture, data, augmentation, optimizer, learning rate and 300 epochs.
  - Each setting has 3 seeds, and accuracy is measured at the final epoch on the test set, which was never used for any choice.
  - All runs were Wulver batch jobs (`experiments/wulver/`, logs included).
  - Limitation: the learning rate was tuned on 7/4 and shared by every ViT.
- **Result (Figure 2, `results/figures/claim.png`):**

| Patch size | Tokens | Test accuracy (%) | GFLOPs/image | GPU-hours/run |
| --- | --- | --- | --- | --- |
| 8x8 | 16 | 87.96 ± 0.29 | 0.128 | 0.95 |
| 4x4 | 64 | 91.57 ± 0.49 | 0.509 | 1.04 |
| 2x2 | 256 | 91.71 ± 0.35 | 2.361 | 3.42 |

  - Going from 8x8 to 4x4 adds 3.61 points.
  - Going from 4x4 to 2x2 adds 0.14 points, within the seed standard deviation (0.35 to 0.49), for 4.64x the FLOPs and 3.3x the GPU time. The returns from finer patches run out.
- **Compute-matched check** (Figure 3, `results/figures/compute.png`).
  - ViT-Lite-7/2 trained for 65 epochs (300 / 4.64), the training FLOPs of 7/4 at 300 epochs, reaches 78.53 ± 1.02%: 13.04 points below 7/4.
  - Two caveats from `experiments/decisions.md` go in the text:
    - the learning rate was not re-tuned for 65 epochs;
    - the match is in FLOPs, not wall-clock time (0.67 h against 1.04 h).
- **Learning curves** (Figure 4, `results/figures/curves.png`):
  - Mixup and CutMix switch off at epoch 175.
  - Plotted against training compute, ViT-Lite-7/2 trails 7/4 at every point.
- **Extension, the conv stem** (also in Figures 2 and 3): 93.08 ± 0.22%.
  - That is +1.51 over 7/4 for 15% more FLOPs, and +1.37 over 7/2 with a quarter of the FLOPs and under a third of the GPU time.
  - It is consistent with Xiao et al. (2021).

## 5. Analysis and Outlook (rubric E)

Rubric: evidence-based verdict on the claim, insight into where the model breaks, quality of the forward-looking discussion.

**5.1 Verdict: the claim holds.**
- ResNet-18 leads by 8.15, 4.54 and 4.39 points at patch sizes 8, 4 and 2, each far above the 1-point margin and at least 9 times the largest seed standard deviation (0.49).
- **Worst-case check:** the lowest ResNet seed (95.99) minus the best ViT seed still leaves 7.85, 4.02 and 3.88 points.
- **Beyond the claim:** the conv stem also stays 3.02 points below (2.66 in the worst case).
- **Robustness to the replication gap:** even adding the 2.00 points by which ViT-Lite-7/4 trails its published value, the gap (2.54) would stay above the margin.

**5.2 Where the model breaks**
- **By class** (Figure 5, `results/figures/per_class.png`):
  - the ViT loses most on animals: cat -9.97, dog -7.97, bird -6.30 and deer -5.83 points, against ResNet-18, for 7/4;
  - it loses least on truck (-1.60) and frog (-1.63);
  - cat and dog swapped is the top confusion for every model;
  - cat -> frog appears among the ViTs' top four confusions (99 cases for 7/4 over 3 seeds) but not among ResNet-18's (`results/figures/figures.md`).
- **By layer** (`results/analysis/`).
  - **Attention distance** (Figure 6):
    - In the first layer, from-scratch attention is close to uniform: 14.7 to 15.1 px against 16.1 to 16.7 px for uniform attention.
    - Layer 2 is the most local layer, at 11.9, 10.7 and 8.5 px.
    - The pretrained ViT-B/16 has local heads from layer 1, down to 0.0 px.
  - **Layer knockout** (Figure 7):
    - Removing the block-1 MLP costs 77.7, 80.4 and 82.5 points.
    - Removing layer-2 attention costs 4.2, 11.3 and 34.0 points for patch sizes 8, 4 and 2.
    - The from-scratch ViT builds its local features in its first two blocks, and depends on them heavily.
  - **Linear probes** (Figure 8):
    - Patch tokens averaged before any block reach 29.6% (7/4), against 49.5% after ResNet-18's stem.
    - The CLS token after block 1 reaches 46.8% (7/4), against 53.5% for ViT-B/16.
- **What the conv stem changes.** It moves local feature extraction out of attention instead of making attention local.
  - Attention gets less local (layer 2: 12.3 px against 10.7).
  - Dependence on the block-1 MLP falls from 80.4 to 55.7 points, and no attention branch costs more than 3.9.
  - Tokens before any block probe at 65.1%.
  - Its lead over 7/4 shrinks from 23.0 points after block 1 to 1.5 at the last block, and ResNet-18's pooled features overtake it in the last third of the network. The remaining 3-point gap opens in the deeper layers.

**5.3 Limitations**
- One dataset at 32 px.
- The ViT learning rate was tuned only on 7/4.
- The ViTs have no stochastic depth or attention dropout (see section 3).
- 3 seeds, and a single seed for ViT-B/16.
- The compute-matched run has the caveats above.
- The analyses use one checkpoint per seed and the first 1,000 or 10,000 test images.

**5.4 Outlook**
- **Next experiments:**
  - Add stochastic depth 0.1 and attention dropout 0.1 to ViT-Lite-7/4 and the conv stem. This tests the replication gap and whether regularization closes part of the 3 points.
  - Try SeqPool instead of the CLS token, and shifted patch tokenization.
  - Train one model across patch sizes (FlexiViT-style) to get the whole accuracy-against-compute curve from a single training run.
- **Applications (forward-looking):**
  - **Small, low-resolution image domains where ImageNet pretraining transfers poorly:** microscopy and histopathology tiles, astronomical cutouts, detector images, low-resolution satellite tiles. There, the choice between a CNN, a conv-stem ViT and a pretrained ViT is real, and our results suggest starting from the conv stem.
  - **Patch size as a deployment knob** on edge devices: the 8x8 model uses 0.128 GFLOPs per image, about 1/9 of ResNet-18's.
  - **Knockout and probes as a cheap diagnostic** of whether a ViT is data-starved: heavy dependence on block 1 and near-uniform first-layer attention flag a model that lacks locality.

## 6. Figures and tables

| # | Content | File | Status |
| --- | --- | --- | --- |
| Figure 1 | Pipeline diagram, patches against conv stem | to draw (TikZ or draw.io) | to do |
| Figure 2 | Accuracy against patch size, ResNet-18 and claim threshold | `results/figures/claim.pdf` | done |
| Figure 3 | Accuracy against training compute and GPU-hours | `results/figures/compute.pdf` | done |
| Figure 4 | Learning curves by epoch and by compute | `results/figures/curves.pdf` | done |
| Figure 5 | Per-class accuracy | `results/figures/per_class.pdf` | done |
| Figure 6 | Mean attention distance by layer | `results/analysis/attention_distance.png` | done |
| Figure 7 | Layer knockout | `results/analysis/knockout.png` | done |
| Figure 8 | Linear probes by layer | `results/analysis/probes.png` | done |
| Figure 9 (optional) | Similarity of learned position embeddings | `demo/demo.ipynb`, section 2, step 3 | done |
| Tables 1 to 3 | Models; learning-rate selection; replication anchors | numbers above | done |

## 7. Video (mini-lecture; it should closely follow the report)

1. **The question** (1 min): ViTs need pretraining; can patch size fix that at 32 px? State the claim.
2. **How a ViT sees an image** (2 min): a screen recording of demo sections 2 and 3, showing the patches, tokens, attention and the conv stem's 13x13 window.
3. **Baseline and replication** (1.5 min): Tables 2 and 3, and the honest gap.
4. **The patch-size study** (2 min): Figures 2 and 3, and the compute-matched run.
5. **Where it breaks** (2.5 min): Figures 5 to 8 and the conv-stem story.
6. **Verdict and outlook** (1 min).

## 8. Schedule

| Date | Milestone |
| --- | --- |
| By Dec 3 | Report draft (LaTeX in `report/`) and video draft |
| By Dec 10 | Final checks: every number against `results/`, links on page 1, PDF built |
| Dec 13 | PDF on Canvas, and the video and report link on the discussion thread |

## References (checked)

- Beyer et al. (2023). FlexiViT: One Model for All Patch Sizes. CVPR. arXiv:2212.08013.
- d'Ascoli et al. (2021). ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases. ICML. arXiv:2103.10697.
- Dosovitskiy et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. arXiv:2010.11929.
- Hassani et al. (2021). Escaping the Big Data Paradigm with Compact Transformers. arXiv:2104.05704.
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR. arXiv:1512.03385.
- Lee, Lee and Song (2021). Vision Transformer for Small-Size Datasets. arXiv:2112.13492.
- Nguyen et al. (2024). An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels. arXiv:2406.09415.
- Raghu et al. (2021). Do Vision Transformers See Like Convolutional Neural Networks? NeurIPS. arXiv:2108.08810.
- Steiner et al. (2022). How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers. TMLR. arXiv:2106.10270.
- Touvron et al. (2021). Training data-efficient image transformers & distillation through attention. ICML. arXiv:2012.12877.
- Wu, Ren, Liao and Grosse (2018). Understanding Short-Horizon Bias in Stochastic Meta-Optimization. ICLR. arXiv:1803.02021.
- Xiao et al. (2021). Early Convolutions Help Transformers See Better. NeurIPS. arXiv:2106.14881.
- Recipe details: the released CCT CIFAR-10 config (`configs/pretrained/cct_7-3x1_cifar10_300epochs.yml`) and the ViT-Lite defaults (`src/vit.py`), both in github.com/SHI-Labs/Compact-Transformers.
