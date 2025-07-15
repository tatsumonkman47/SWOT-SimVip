#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=SIMVIP_ARRAY
#SBATCH --output=sbatch_output_logs/simvip_grad_loss_%A_%a.out
#SBATCH --error=sbatch_output_logs/simvip_grad_loss_%A_%a.err
#SBATCH --mail-user=tm3076@nyu.edu
#SBATCH --array=0-7   # <-- adjust range to match the number of configs

# List of config names (without .yaml)
CONFIG_NAMES=(
    "001_C2_config_05rho"
    "002_C2_config_07rho"
    "003_C2_10rho"
    "004_C2_config_02rho"
    "011_C1_config_05rho"
    "012_C1_config_07rho"
    "013_C1_config_10rho"
    "014_C1_config_02rho"
)

# Select the config corresponding to the current array index
CONFIG_NAME=${CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running job with config: ${CONFIG_NAME}.yaml"

module purge

singularity exec --nv \
    --overlay /scratch/tm3076/test_python_env/my_conda.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate swot_python12_pytorch; python base_simvip_template.py --config-name=${CONFIG_NAME}"