#!/bin/bash
#SBATCH --job-name=tbt-a4
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=0-08:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Usage: sbatch run_sol.sbatch                (all strategies)
#        sbatch run_sol.sbatch s4,s5          (subset)
# The LLM runs on the RC gateway, so no GPU is needed; CPUs are for
# executing and validating generated programs in parallel.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load mamba/latest
source activate a4

STRATEGIES="${1:-s1,s2,s4,s5}"

echo "job $SLURM_JOB_ID on $(hostname), strategies=$STRATEGIES, started $(date)"
python check_harness.py
python run_benchmark.py --strategy "$STRATEGIES" --cases all --workers 10 --llm-concurrency 6
python summarize.py || true
echo "finished $(date)"
