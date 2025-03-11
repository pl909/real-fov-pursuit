#!/usr/bin/bash
#SBATCH --job-name=pursuit-evasion
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64g
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --partition=athena-genai
#SBATCH --account=pl217
#SBATCH --gres=gpu:1

# Load module if needed (uncomment if your cluster requires this)
# module load anaconda3

# Proper way to activate conda in batch scripts
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drones

# Print conda and environment info for debugging
echo "Using conda at: $(which conda)"
echo "Python interpreter: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ $(python -c 'import torch; print(torch.cuda.is_available())') = "True" ]; then
    echo "GPU device name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Run the experiment
python learn_pursuit.py

# Cleanup
conda deactivate