#!/bin/bash
# MuSE translation job (Della)
#
# Usage: sbatch translate-della.slurm <model> <input> <output>
#   model   — model identifier: hymt | madlad | nllb | gemma
#   input   — path to input JSONL file
#   output  — path to write translation output JSONL
#
# Placeholder variables to update before submitting:
#   FACULTY_NETID  — Slurm account (faculty collaborator's netid)
#   YOUR_NETID     — your Princeton netid
#
# To run on GPU, uncomment the ##SBATCH lines below and remove --mem-per-cpu.
#
#SBATCH --job-name=muse-translate
#SBATCH --account=FACULTY_NETID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G               # CPU only — remove this line for GPU jobs
##SBATCH --gres=gpu:1                   # GPU only — uncomment to request a GPU
##SBATCH --partition=mig                # GPU only — default MIG slice; use 'gpu' for a full A100
#SBATCH --time=00:15:00                 # Wall time limit — increase for large corpora
#SBATCH --output=logs/%x_%j.out        # Logs are written to logs/ relative to the working directory
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
NETID=YOUR_NETID
REPO=/scratch/gpfs/CDHRSE/${NETID}/muse

module purge
# uv is not available as a module, so we load anaconda to get conda
module load anaconda3/2025.12
# Activate the muse conda environment (contains uv; see della-instructions.md for setup)
conda activate muse

export HF_HOME=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache
export HF_HUB_CACHE=/scratch/gpfs/CDHRSE/${NETID}/huggingface-cache/hub
export HF_HUB_OFFLINE=1

# ---------------------------------------------------------------------------
# Validate arguments
# ---------------------------------------------------------------------------
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: sbatch translate-della.slurm <model> <input> <output>"
    echo "  model   — hymt | madlad | nllb | gemma"
    echo "  input   — path to input JSONL file"
    echo "  output  — path to write translation output JSONL"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run translation
# ---------------------------------------------------------------------------
cd ${REPO}

uv run src/muse/translation/translate_corpus.py $1 $2 $3
