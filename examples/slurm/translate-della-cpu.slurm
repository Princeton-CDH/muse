#!/bin/bash
# MuSE translation job — CPU (Della)
#
# Usage: sbatch translate-della-cpu.slurm <model> <input> <output>
#   model   — model identifier: hymt | madlad | nllb | gemma
#   input   — path to input JSONL file
#   output  — path to write translation output JSONL
#
# Placeholder variables to update before submitting:
#   FACULTY_NETID  — Slurm account (faculty collaborator's netid)
#   YOUR_NETID     — your Princeton netid
#
#SBATCH --job-name=muse-translate
#SBATCH --account=FACULTY_NETID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G               # 10G is sufficient for 1.8B–4B models
#SBATCH --time=00:15:00                 # Wall time limit — increase for large corpora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
NETID=YOUR_NETID
REPO=/scratch/gpfs/CDHRSE/${NETID}/muse

module purge
module load anaconda3/2025.12
conda activate muse

export HF_HOME=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache
export HF_HUB_CACHE=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache/hub
export HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Run translation
# ---------------------------------------------------------------------------
cd ${REPO}

uv run src/muse/translation/translate_corpus.py $1 $2 $3 --verbose
