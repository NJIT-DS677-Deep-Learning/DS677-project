#!/bin/bash -l
#SBATCH --job-name=demo
#SBATCH --output=experiments/wulver/logs/%x.%j.out
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-rv44
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00

# Exports the seed-0 weights the demo uses (demo/weights/) and executes demo/demo.ipynb in place,
# so the notebook in the repository shows real outputs: full test set, GPU, ViT-B/16 included.
# Submit from the repository root:  sbatch experiments/wulver/demo.sbatch
source experiments/wulver/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi -L
jupyter kernelspec list

echo "== export weights"
python experiments/export_weights.py

echo "== execute the demo notebook"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 demo/demo.ipynb

echo "== done"
ls -la demo demo/weights
