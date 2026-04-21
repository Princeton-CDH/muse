# Running MuSE Translation Jobs on Della

Refer to Princeton RC's [Della documentation](https://researchcomputing.princeton.edu/systems/della) for further information about using Della.

## Prerequisites

- A Princeton HPC account with access to Della — request access through the [Research Computing portal](https://researchcomputing.princeton.edu/get-started/get-account)
- Membership in the `CDHRSE` group to access `/scratch/gpfs/CDHRSE/`
- The faculty collaborator's netid to use as the Slurm `--account`

## Scratch Storage

If you haven't already, create your personal scratch directory witihin the CDH RSE team's scratch at `/scratch/gpfs/CDHRSE/<netid>/`.

A few things to know about this directory:

- It is primarily for storing job input and output files as well as any other intermediary or supplmentary files.
- It is **not backed up**, so make sure to save any important outputs to TigerData.
- Unlike `/tmp` and other directories within `/scratch`, your personal scratch directory is **not purged**. This means that files persist across sessions, but also it is your responsibility to remove files to avoid hitting group quota limits.
- Large model files and corpora should live here rather than in your home directory, which has a much smaller quota. For more on data storage quotas see [Princeton RC's documentation on data storage](https://researchcomputing.princeton.edu/support/knowledge-base/data-storage#Filesystem-Details).

## Setup

### Set up the muse working directory

Clone the repo into your scratch space:

```bash
cd /scratch/gpfs/CDHRSE/<netid>
git clone <repo-url> muse
```

Create the logs directory that the Slurm script writes to:

```bash
cd muse
mkdir -p logs
```

### Create the conda environment

Della's module system does not include `uv`, so we use `conda` as a thin wrapper solely to make `uv` available. The actual Python environment and dependencies are managed by `uv`. Create the environment once on a login node:

```bash
module purge
module load anaconda3/2025.12
conda create -n muse python=3.12 -y
conda activate muse
pip install uv
uv sync
```

NOTE: We tried using a project-specific conda environment with all dependencies managed by conda, but ran into compatibility issues (see [issue #55](https://github.com/Princeton-CDH/muse/issues/55)). Our current workaround is to create a minimal conda env that installs `uv`, which then manages everything else.

### Set up the HuggingFace cache

Compute nodes have no internet access, so models must be cached on a login node before submitting jobs. The cache should live in your personal scratch. Be sure to set the appropriate HuggingFace environment variables so that HuggingFace knows the location of your cache:

```bash
export HF_HOME=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache
export HF_HUB_CACHE=/scratch/gpfs/CDHRSE/<netid>/huggingface-cache/hub
```

To populate the cache, copy the models over from your local environment. **Do not obtain them by running code that loads these models on the head node.** Loading HuggingFace models is a non-trivial task that requires a lot of RAM, especially for larger models.

## Submitting a Job

The example script located at [`examples/slurm/translate-della.slurm`](https://github.com/Princeton-CDH/muse/blob/develop/examples/slurm/translate-della.slurm) runs [`translate_corpus`](https://github.com/Princeton-CDH/muse/blob/develop/src/muse/translation/translate_corpus.py)). It expects the three input arguments that `translate_corpus` requires:

```bash
sbatch examples/slurm/translate-della.slurm <model> <input> <output>
```

For example:

```bash
sbatch examples/slurm/translate-della.slurm hymt input.jsonl output.jsonl
```

Before submitting this job, update the two placeholder variables:

- `FACULTY_NETID` — the Slurm account (faculty collaborator's netid), used for `--account`
- `YOUR_NETID` — your Princeton netid, used to construct scratch paths

### Script configuration

The script is configured for a **CPU job** by default with the following slurm directives:

- `--cpus-per-task=1`: single CPU; the translation models are not parallelised across CPUs
- `--mem-per-cpu=10G`: 10G is sufficient for our working models with 1.8B–4B parameters
- `--time=00:15:00`: 15-minute wall time limit; increase this for large corpora

### Running on GPU

GPU jobs run ~14x faster than CPU for our working models. To switch to GPU:

1. Uncomment `##SBATCH --partition=mig` in the script.
2. Remove or comment out `--mem-per-cpu=10G`: All mig jobs are assigned a CPU memory of 32GB.

The directive `##SBATCH --partition=mig`, means our job will be allocated to a MIG GPU which has 10GB of GPU memory, 1 CPU core, and 32GB of CPU memory. If your job needs multiple GPUs, CPUs or additional memory, you will need to use a different slurm directive. See the [Della documentation](https://researchcomputing.princeton.edu/systems/della#GPU-Jobs) for information on other GPU options and their corresponding slurm directives.

All HuggingFace models are loaded with `device_map="auto"`, so they use the GPU automatically if one is available.

## Logs

Job stdout and stderr are configured to be written to the `logs` directory within the `muse` repo.
Stdout is written to the logfile named `<job-name>_<job-id>.out` and stderr is written to `<job-name>_<job-id>.err`. These files will update while the job is running, however, there will likely be a lag due to python's buffering (consider running python with `-u` to make stdout and stderr unbuffered).

## Useful Commands

- Check job status: `squeue -u <netid>`
- Check efficiency after job completes: `jobstats <jobid>`
- Check your current data storage and storage limits: `checkquota`
- Inspect the directories in your personal scratch that use the most space:
  ```bash
  du -h ---max-depth=1 /scratch/gpfs/<ResearchGroup>/<YourNetID> | sort -hr
  ```
