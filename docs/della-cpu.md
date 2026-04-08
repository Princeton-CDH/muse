# Running MuSE Translation Jobs on Della (CPU)

For general Della setup and CPU job guidance, see the [Della documentation](https://researchcomputing.princeton.edu/systems/della).

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

Edit `examples/slurm/translate-della-cpu.slurm` and update `YOUR_NETID`, `YOUR_ACCOUNT`, `INPUT`, `OUTPUT`, and `MODEL`, then:

```bash
sbatch examples/slurm/translate-della-cpu.slurm
```

## Useful Commands

```bash
# Check job status
squeue -u <netid>

# Check CPU efficiency after job completes
jobstats <jobid>

# Pull latest code on a login node
git pull && uv sync

# Update HuggingFace cache (login node only)
export HF_HOME=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('tencent/HY-MT1.5-1.8B'); \
    AutoModelForCausalLM.from_pretrained('tencent/HY-MT1.5-1.8B')"
```
