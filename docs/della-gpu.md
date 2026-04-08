# Running MuSE Translation Jobs on Della (GPU)

For general Della setup and GPU job guidance, see the [Della documentation](https://researchcomputing.princeton.edu/systems/della#GPU-Jobs).

## Prerequisites

- A Princeton HPC account with access to Della
- Access to `/scratch/gpfs/CDHRSE/`
- HuggingFace models pre-cached at `/scratch/gpfs/CDHRSE/<netid>/huggingface-cache` (compute nodes have no internet — download models on a login node first)

## Setup

Clone the repo to scratch and create the logs directory:

```bash
cd /scratch/gpfs/CDHRSE/<netid>/muse
git clone <repo-url> muse && cd muse
mkdir -p logs
```

## Submitting a Job

Edit `examples/slurm/translate-della-gpu.slurm` and update `YOUR_NETID`, `YOUR_ACCOUNT`, `INPUT`, `OUTPUT`, and `MODEL`, then:

```bash
sbatch examples/slurm/translate-della-gpu.slurm
```

## GPU Support Note

All HuggingFace models in `translate.py` are loaded with `device_map="auto"`, so they will use the GPU automatically when `--gres=gpu:1` is present in the Slurm script. No additional configuration is needed.

## Useful Commands

```bash
# Check job status
squeue -u <netid>

# Check GPU efficiency after job completes
jobstats <jobid>

# Pull latest code on a login node
git pull && uv sync

# Update HuggingFace cache (login node only)
export HF_HOME=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('tencent/HY-MT1.5-1.8B'); \
    AutoModelForCausalLM.from_pretrained('tencent/HY-MT1.5-1.8B')"
```
