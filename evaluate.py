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

# Ignite
from ignite.engine import Engine
from ignite.metrics import Loss, SSIM, FID, MeanAbsoluteError
from ignite.utils import manual_seed
from ignite.handlers import Timer

from dataloader import get_data_loader

torch.manual_seed(42)

def setup_metrics(opts):
    # Define & Initializer metrics in dictionary
    metrics=dict()

    ### TODO: Add more metrics here
    if opts.fid:
        metrics["fid"] = FID()
    if opts.ssim:
        # For tmed output (HDR)
        metrics["ssim"] = SSIM(data_range=16.)
    if opts.l1:
        metrics["l1"] = MeanAbsoluteError()

    # Reset to double check
    for k in metrics:
        metrics[k].reset()

    return metrics

def evaluate_metrics(opts, metrics):
    dataloader = get_data_loader(path_pred=f"{opts.predictions_dir}", 
                            path_gt=f"{opts.targets_dir}",
                            batch_size=opts.batch_size)
    # Fine grained control -> enable rolling calcs
    for i,  (y_pred, y) in enumerate(dataloader):
        print(f"Batch {i+1} :: y_pred.shape: {y_pred.shape} | y.shape: {y.shape}")
        for metric in metrics:
            # print(f"Before: {metric} = {metrics[metric]}") 
            metrics[metric].update((y_pred, y))
    
    for metric in metrics:
        metrics[metric] = metrics[metric].compute()
    save_eval_results(opts.exp_name, metrics)

def save_eval_results(RUN_NAME: str, metrics: dict):
    with open(f"results/{RUN_NAME}.txt", "a") as f:
        f.write("Evaluation Metrics:\n")
        for k,v in metrics.items():
            f.write(f"{k} = {v}\n")

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
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help="Batch Size.")
    
    # Metric Options
    parser.add_argument('--fid', action='store_true', default=True)
    parser.add_argument('--ssim', action='store_true', default=True)
    parser.add_argument('--l1', action='store_true', default=True)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = Path(f"eval_run-{datetime.now().strftime('%Y%m%d_%H_%M_%S')}/")

    assert args.targets_dir is not None, f"[!] Targets' Directory not specified. Usage: ./evaluate.py -td <target_dir>"
    assert args.predictions_dir is not None, f"[!] Predictions' Directory not specified. Usage: ./evaluate.py -pd <preds_dir>"

    run(args)

# ./evaluate.py -en test_run_1 -td datasets/dummy_verification_ds/targets/ -pd datasets/dummy_verification_ds/predictions/ -bs 2 --fid --ssim --l1