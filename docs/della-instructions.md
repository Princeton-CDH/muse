# Running MuSE Translation Jobs on Della

For general Della documentation, see the [Princeton Research Computing Della page](https://researchcomputing.princeton.edu/systems/della).

## Prerequisites

- A Princeton HPC account with access to Della — request access through the [Research Computing portal](https://researchcomputing.princeton.edu/get-started/request-account)
- Membership in the `CDHRSE` group to access `/scratch/gpfs/CDHRSE/` — ask a current CDH RSE to add you
- The faculty collaborator's netid to use as the Slurm `--account`

## Scratch Storage

CDH RSE files live at `/scratch/gpfs/CDHRSE/<netid>/`. A few things to know about this space:

- It is **not backed up** — do not store anything you cannot reproduce
- Unlike `/tmp` and other scratch areas, `/scratch/gpfs` is **not purged** on a schedule, so files persist across sessions
- Large model files and corpora should live here rather than in your home directory, which has a much smaller quota

## Setup

### Clone the repo

```bash
cd /scratch/gpfs/CDHRSE/<netid>
git clone <repo-url> muse && cd muse
mkdir -p logs
```

### Create the conda environment

The Slurm script activates a conda environment named `muse`. Create it once on a login node:

```bash
module purge
module load anaconda3/2025.12
conda create -n muse python=3.12 -y
conda activate muse
pip install uv
uv sync
```

### Set up the HuggingFace cache

Compute nodes have no internet access, so models must be cached on a login node before submitting jobs. The cache should live in scratch:

```bash
export HF_HOME=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache
export HF_HUB_CACHE=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache/hub
```

To populate the cache, either download models directly on a login node:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('tencent/HY-MT1.5-1.8B')
AutoModelForCausalLM.from_pretrained('tencent/HY-MT1.5-1.8B')
"
```

Or copy your local cache from your dev machine (faster if you already have the models):

```bash
scp -r ~/.cache/huggingface/ <netid>@della.princeton.edu:/scratch/gpfs/CDHRSE/<netid>/huggingface-cache
```

## Submitting a Job

The example script is at `examples/slurm/translate-della.slurm`. It accepts three positional arguments:

```bash
sbatch examples/slurm/translate-della.slurm <model> <input> <output>
```

For example:

```bash
sbatch examples/slurm/translate-della.slurm hymt input.jsonl output.jsonl
```

Before submitting, update the two placeholder variables at the top of the script:
- `FACULTY_NETID` — the Slurm account (faculty collaborator's netid), used for `--account`
- `YOUR_NETID` — your Princeton netid, used to construct scratch paths

### Script configuration

The script is configured for a **CPU job** by default:

- `--cpus-per-task=1` — single CPU; the translation models are not parallelised across CPUs
- `--mem-per-cpu=10G` — 10G is sufficient for the 1.8B–4B parameter models
- `--time=00:15:00` — 15-minute wall time limit; increase this for large corpora

### Running on GPU

GPU jobs run ~14x faster than CPU for the 1.8B–4B models. To switch to GPU:

1. Uncomment `##SBATCH --gres=gpu:1` in the script
2. Remove or comment out `--mem-per-cpu` — GPU memory allocation is pre-defined by the partition and cannot be set manually

By default, `--gres=gpu:1` allocates a MIG slice of an A100 (the `mig` partition). If you need a full A100, also add `--partition=gpu`.

All HuggingFace models are loaded with `device_map="auto"`, so they use the GPU automatically when a GPU is allocated — no code changes needed.

## Logs

Job stdout and stderr are written to `logs/` in the repo directory, named `<job-name>_<job-id>.out` and `.err`. Check them after a job completes:

```bash
cat logs/muse-translate_<jobid>.out
cat logs/muse-translate_<jobid>.err
```

## Useful Commands

```bash
# Check job status
squeue -u <netid>

# Check efficiency after job completes
jobstats <jobid>

# Pull latest code and sync dependencies (login node)
git pull && uv sync
```
