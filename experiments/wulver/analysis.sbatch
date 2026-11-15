#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --output=experiments/wulver/logs/%x.%j.out
#SBATCH --partition=course_gpu
#SBATCH --account=2026-fall-ds-677-ikoutis-rv44
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
# Layer analyses for research question 3, on the finished checkpoints (forward passes only, no
# training). Writes results/analysis/{attention_distance,knockout,probes}.{json,md,png}.
# Submit from the repo root:  sbatch experiments/wulver/analysis.sbatch
source experiments/wulver/env.sh
export HF_HUB_OFFLINE=1   # the ImageNet-21k checkpoint is already cached (prepare_data.py --cache-vit)
nvidia-smi -L
echo "== attention distance";  python src/analysis/attention_distance.py --n 1000
echo "== layer knockout";      python src/analysis/knockout.py
echo "== linear probes";       python src/analysis/probes.py --n-train 10000 --n-test 10000
echo "== done"; ls -la results/analysis
