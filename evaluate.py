#!/usr/bin/python

import os
from pathlib import Path

# Base
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import tqdm

# Torch-vision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# Ignite
from ignite.engine import Engine
from ignite.metrics import Loss, SSIM, FID, MeanAbsoluteError
from ignite.utils import manual_seed
from ignite.handlers import Timer

torch.manual_seed(42)

def setup_metrics(opts):
    # Define & Initializer metrics in dictionary
    metrics=dict()

    ### TODO: Add more metrics here
    if opts.fid:
        metrics["fid"] = FID()
    if opts.ssim:
        metrics["ssim"] = SSIM()
    if opts.l1:
        metrics["l1"] = L1()

    # Reset to double check
    for k in metrics:
        metrics[k].reset()

    return metrics

def evaluate_metrics(opts, metrics):
    # Call Dataloader 
    # data = DataLoader()
    # Fine grained control -> enable rolling calcs
    for y_pred, y in data:
        for metric, metric_value in metrics.items(): 
            metrics[metric] = metric_value.update((y_pred, y))
    
    save_eval_results(opts.exp_name, metrics)

def save_eval_results(RUN_NAME: str, metrics: dict):
    with open(f"results_{RUN_NAME}") as f:
        f.write("Evaluation Metrics:")
        for k,v in metrics.items():
            f.write(f"{k} = {v}")

def run(opts):
    metrics = setup_metrics(opts)
    evaluate_metrics(opts, metrics)
    pass

if __name__ == "__main__":
    parser = ArgumentParser(prog = 'Evaluation Metrics',
                    description = 'Runs Evaluation Metrics on results. (Stand-alone evaluator)',
                    epilog = 'Rights reserved by professor Jean-Francois Lalonde')
    parser.add_argument('-en','--exp_name', type=str, default=None, help="Name of experiment. Used for saving results.")
    parser.add_argument('-td', '--targets_dir', type=str, default=None, help="GT directory.")
    parser.add_argument('-pd', '--predictions_dir', type=str, default=None, help="Predictions' dir.")
    # Metric Options
    parser.add_argument('--fid', action='store_true')
    parser.add_argument('--ssim', action='store_true')
    parser.add_argument('--l1', action='store_true')

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = Path(f"evaluation_runs/logs-{datetime.now().strftime('%Y%m%d_%H_%M_%S')}/")

    assert args.targets_dir is not None, f"[!] Targets' Directory not specified. Usage: ./evaluate.py -td <target_dir>"
    assert args.predictions_dir is not None, f"[!] Predictions' Directory not specified. Usage: ./evaluate.py -pd <preds_dir>"

    run(args)