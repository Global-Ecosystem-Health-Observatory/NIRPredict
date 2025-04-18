import os
import torch
import argparse

from nirpredict.data.loader import load_data

from nirpredict.modeling.model import build_model, load_best_weights
from nirpredict.modeling.trainer import train
from nirpredict.modeling.evaluator import evaluate

from nirpredict.utils.config import setup
from nirpredict.utils.logger import get_logger

logger = get_logger(__name__)


def run(conf):
    logger.info("Starting the NIR Predictor.")

    assert os.path.exists(conf.data_folder), f"[ERROR] Data folder {conf.data_folder} does not exist."

    if not os.path.exists(conf.output_dir):
        os.makedirs(conf.output_dir)
        logger.info(f"Created output directory: {conf.output_dir}")
    else:
        logger.info(f"Output directory already exists: {conf.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Building the NIR model...")
    nir_model, criterion, optimizer = build_model(device, lr=conf.learning_rate, outdir=conf.output_dir)
    logger.info("Model built successfully.")
    
    logger.info("Loading data...")
    train_nir_loader, val_nir_loader, test_nir_loader = load_data(conf)
    logger.info("Data loaded successfully.")
    
    logger.info("Starting training...")
    train(nir_model, train_nir_loader, val_nir_loader, optimizer, criterion, device, num_epochs=conf.epochs, outdir=conf.output_dir)
    logger.info("Training completed.")
    
    logger.info("Starting evaluation...")
    evaluate(nir_model, test_nir_loader, criterion, device)
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration setup for network.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}...")
    conf = setup(args.config)
    logger.info("Configuration loaded successfully.")

    logger.info(f"Data folder set to {conf.data_folder}.")
    
    run(conf)


'''

Usage:

python3 -m nirpredict.main ./configs/nirpredict.txt

- For Puhti

export TREEMORT_VENV_PATH="/projappl/project_2004205/rahmanan/venv"
export TREEMORT_REPO_PATH="/users/rahmanan/TreeMort"

sbatch --export=ALL,CONFIG_PATH="$TREEMORT_REPO_PATH/configs/nirpredict.txt" $TREEMORT_REPO_PATH/scripts/run_nirpredict.sh

'''