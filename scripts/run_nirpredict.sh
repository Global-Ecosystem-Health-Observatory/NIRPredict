#!/bin/bash
#SBATCH --job-name=nir-predict                     # Job name
#SBATCH --account=project_2004205                  # Project account
#SBATCH --output=output/stdout/%A_%a.out           # Output log
#SBATCH --error=output/stderr/%A_%a.err            # Error log
#SBATCH --ntasks=1                                 # Number of tasks (1 process)
#SBATCH --cpus-per-task=4                          # Number of CPU cores per task
#SBATCH --time=08:00:00                            # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                            # Partition to submit to
#SBATCH --gres=gpu:v100:1                          # Number and type of GPU
#SBATCH --mem-per-cpu=8G                           # Memory per GPU in GB

# Usage:
# export TREEMORT_VENV_PATH="/custom/path/to/venv"
# sbatch --export=ALL,CONFIG_PATH="/custom/path/to/config" run_nirpredict.sh

MODULE_NAME="pytorch/2.3"

TREEMORT_VENV_PATH="${TREEMORT_VENV_PATH:-/projappl/project_2004205/rahmanan/venv}"

echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

if [ -d "$TREEMORT_VENV_PATH" ]; then
    echo "[INFO] Activating virtual environment at $TREEMORT_VENV_PATH"
    source "$TREEMORT_VENV_PATH/bin/activate"
else
    echo "[ERROR] Virtual environment not found at $TREEMORT_VENV_PATH"
    exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
    echo "[ERROR] CONFIG_PATH variable is not set. Please provide a config path using --export."
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config file not found at $CONFIG_PATH"
    exit 1
fi

echo "[INFO] Starting NIR Predictor with the following settings:"
echo "       Config file: $CONFIG_PATH"
echo "       CPUs per task: $SLURM_CPUS_PER_TASK"
echo "       Memory per CPU: $SLURM_MEM_PER_CPU MB"
echo "       Job time limit: $SLURM_TIMELIMIT"

srun python3 -m nirpredict.main "$CONFIG_PATH"

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
    echo "[ERROR] NIR Predictor failed with exit status $EXIT_STATUS"
else
    echo "[INFO] NIR Predictor completed successfully"
fi

exit $EXIT_STATUS