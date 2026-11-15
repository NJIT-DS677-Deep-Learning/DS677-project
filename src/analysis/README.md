# Layer analyses (research question 3)

All three scripts load the final checkpoints of finished runs (`results/<run>/seed<k>/last.pt`) and use forward passes only, so nothing is retrained. Run them together on Wulver with `sbatch experiments/wulver/analysis.sbatch`; each writes a JSON file, a markdown table and a figure to `results/analysis/`.

| Script | Question | What it measures |
| --- | --- | --- |
| `attention_distance.py` | Does a ViT trained from scratch learn local attention in its early layers? | For every head in every layer, the attention-weighted mean distance between a patch and the patches it attends to, over 1,000 test images, in original CIFAR pixels (the measure in Figure 7 of Dosovitskiy et al., 2021). Compared with uniform attention and with ViT-B/16 before and after fine-tuning. |
| `probes.py` | Where does class information appear? | Logistic-regression test accuracy on frozen features after each block: the CLS token and the mean of the patch tokens for ViTs, globally pooled feature maps for ResNet-18. |
| `knockout.py` | Which layers does the network depend on? | Test accuracy drop when one block is removed at test time, and for ViTs also when only its attention branch or only its MLP branch is removed. |

`common.py` holds the shared loading code and the fixed color for each model, which is the same in every figure. The choices in each script's docstring (which tokens count, how distances are normalised, the probe settings) were fixed before any real results were analysed.

Tested on CPU with fake data and tiny one-epoch models: removing a whole ViT block gives exactly the same output as removing its attention and MLP branches together, and a saved ViT-B/16 checkpoint reloads without downloading anything.
