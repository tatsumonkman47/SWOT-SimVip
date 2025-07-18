#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20            # Increased from 4
#SBATCH --time=12:00:00               # Reduced from 24h if possible
#SBATCH --mem=64GB                    # Increased from 32GB
#SBATCH --gres=gpu:1                  # Consider adding GPU type
#SBATCH --constraint=a100|rtx8000     # Specific GPU type if available
#SBATCH --gres-flags=enforce-binding  # Exclusive GPU access
#SBATCH --job-name=SIMVIP_ARRAY
#SBATCH --output=sbatch_output_logs/simvip_grad_loss_%A_%a.out
#SBATCH --error=sbatch_output_logs/simvip_grad_loss_%A_%a.err
#SBATCH --mail-user=tm3076@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-18

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=0  # For better async execution


# List of config names (without .yaml)
CONFIG_NAMES=(
    "001_C2_config_05rho"
    "002_C2_config_07rho"
    "003_C2_10rho"
    "004_C2_config_02rho"
    "005_C2_config_00rho"

    "011_C1_config_05rho"
    "012_C1_config_07rho"
    "013_C1_config_10rho"
    "014_C1_config_02rho"
    "015_C1_config_00rho"

    "0X1_C1_10rho_alpha01_test"
    "0X2_C1_10rho_alpha20_test"
    "0X3_C1_10rho_lr1e-6_test"
    "0X4_C1_10rho_lr1e-7_test"

    "021_C3_config_05rho"
    "022_C3_config_07rho"
    "023_C3_config_10rho"
    "024_C3_config_02rho"
    "025_C3_config_00rho"

)

# Select the config corresponding to the current array index
CONFIG_NAME=${CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running job with config: ${CONFIG_NAME}.yaml"

module purge

singularity exec --nv \
    --overlay /scratch/tm3076/test_python_env/my_conda.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate swot_python12_pytorch; python base_simvip_template.py --config-name=${CONFIG_NAME}"
