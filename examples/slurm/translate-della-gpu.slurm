#!/bin/bash
#SBATCH --job-name=muse-translate        # Job name shown in squeue
#SBATCH --account=YOUR_ACCOUNT           # Slurm account (faculty collaborator's account)
#SBATCH --nodes=1                        # Single node job
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=1               # CPUs for data loading / tokenization
#SBATCH --mem-per-cpu=10G               # Memory per CPU (10G is sufficient for 1.8B–4B models)
#SBATCH --gres=gpu:1                    # Request one GPU (A100 on della-gpu partition)
#SBATCH --time=00:15:00                 # Wall time limit (HH:MM:SS) — increase for large corpora
#SBATCH --output=logs/%x_%j.out        # stdout log: logs/<job-name>_<job-id>.out
#SBATCH --error=logs/%x_%j.err         # stderr log: logs/<job-name>_<job-id>.err
#SBATCH --mail-type=END,FAIL            # Email on job end or failure
#SBATCH --mail-user=YOUR_NETID@princeton.edu

# ---------------------------------------------------------------------------
# Paths — update these before submitting
# ---------------------------------------------------------------------------
NETID=YOUR_NETID
REPO=/scratch/gpfs/CDHRSE/${NETID}/muse/muse
INPUT=${REPO}/../data/input.jsonl        # Path to input parallel corpus JSONL
OUTPUT=${REPO}/../data/output.jsonl      # Path to write translation output JSONL
MODEL=hymt                               # Model identifier: hymt | madlad | nllb | gemma

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge                        # Clear any modules inherited from the login shell
module load anaconda3/2025.12       # Load the Anaconda module to make conda available
conda activate muse                 # Activate the muse conda environment

# Point HuggingFace to the shared offline cache — compute nodes have no internet
export HF_HOME=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache
export HF_HUB_CACHE=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache/hub
export HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Run translation
# ---------------------------------------------------------------------------
cd ${REPO}

uv run src/muse/translation/translate_corpus.py \
    ${MODEL} \
    ${INPUT} \
    ${OUTPUT} \
    --verbose
