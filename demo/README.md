# Demo

`demo.ipynb` follows CIFAR-10 test images through the trained models until each one outputs a class. It covers sections A to C of the report (model, baseline, replication anchor) and the patch-size perturbation.

1. **The input:** a 32x32 colour image and how it is normalised.
2. **ViT-Lite-7, step by step:**
   - patches;
   - patch embeddings;
   - CLS token and position embeddings;
   - the 7 transformer blocks;
   - the classification head;
   - where the attention looks.

   Every step is computed by hand and checked against the Hugging Face model.
3. **Patch size and the conv stem:** the same image cut into 16, 64 or 256 tokens, and the conv stem's overlapping 13x13 windows.
4. **The baseline:** the same image through ResNet-18.
5. **Replication anchor and claim check:**
   - test-set accuracy of every model;
   - the training runs;
   - the published numbers;
   - the claim's 1-point margin.
6. **Try your own image:** `classify("photo.jpg")`.

## Running it

- **Google Colab:** [open the notebook](https://colab.research.google.com/github/NJIT-DS677-Deep-Learning/DS677-project/blob/main/demo/demo.ipynb) and run all cells. The first cell clones the repository, and CIFAR-10 is downloaded from the Hugging Face Hub. A GPU is optional. On a CPU, section 5 uses 2,000 test images unless `N_EVAL` is changed.
- **Locally:** from the repository, run `jupyter lab demo/demo.ipynb`. It needs torch, torchvision, transformers 5.x, safetensors, pandas and matplotlib.
- **Wulver:** run `sbatch experiments/wulver/demo.sbatch`. This exports the weights and executes the notebook in place on the full test set. The copy in the repository was produced this way.

## Weights

`weights/` holds seed 0 of every from-scratch model (ResNet-18, ViT-Lite-7/8, 7/4, 7/2 and the conv stem):

- **Format:** float16 safetensors, written by `python experiments/export_weights.py` from the training checkpoints.
- **Size:** about 52 MB in total.
- **Manifest:** `weights/manifest.json` records each file's training config, its SHA-256 and the test accuracy its run logged.

The notebook reports its own measured accuracy next to the logged number.

The fine-tuned ViT-B/16 (86M parameters) is too large for the repository. The notebook evaluates it only where its training checkpoint exists, on Wulver.
