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
from ignite.metrics import Loss, SSIM, FID, MeanAbsoluteError, InceptionScore
from ignite.utils import manual_seed
from ignite.handlers import Timer

from dataloader import get_data_loader

# Specific Losses like lpips?
import lpips
# from lpipBase import PerceptualLoss ---> Can live without this too ;)

torch.manual_seed(42)

# TODO: Has to be global. Find a better way to initialize it:
lpips_vgg = lpips.LPIPS(net='vgg')

def evaluator_lpips_loss(y_pred, y):
    loss = lpips_vgg.forward(y_pred, y)
    return loss.mean()

def setup_metrics(opts):
    # Define & Initializer metrics in dictionary
    metrics=dict()

    ### TODO: Add more metrics here
    if opts.fid:
        metrics["fid"] = FID()
    if opts.ssim:
        # For tmed output (HDR)
        metrics["ssim"] = SSIM(data_range=1.0)
    if opts.l1:
        metrics["l1"] = MeanAbsoluteError()
    if opts.iscore:
        metrics["is"] = InceptionScore()
    if opts.lpips:
        metrics["lpips"] = Loss(evaluator_lpips_loss)

    # Reset to double check
    for k in metrics:
        metrics[k].reset()

    return metrics

def evaluate_metrics(opts, metrics):
    dataloader = get_data_loader(path_pred=f"{opts.predictions_dir}", 
                            path_gt=f"{opts.targets_dir}",
                            batch_size=opts.batch_size)
    # Fine grained control -> enable rolling calcs
    for i,  (y_pred, y) in tqdm(enumerate(dataloader)):
        print(f"\nBatch {i+1} :: y_pred.shape: {y_pred.shape} | y.shape: {y.shape}")
        for metric in metrics:
            if metric is 'is':
                metrics[metric].update(y_pred)
            else:
                metrics[metric].update((y_pred, y))
            # print(f"Before: {metric} = {metrics[metric]}") 
    
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
    parser.add_argument('--iscore', action='store_true', default=True)
    parser.add_argument('--lpips', action='store_true', default=True)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = Path(f"eval_run-{datetime.now().strftime('%Y%m%d_%H_%M_%S')}/")

    assert args.targets_dir is not None, f"[!] Targets' Directory not specified. Usage: ./evaluate.py -td <target_dir>"
    assert args.predictions_dir is not None, f"[!] Predictions' Directory not specified. Usage: ./evaluate.py -pd <preds_dir>"

    run(args)

# ./evaluate.py -en test_run_4 -td datasets/dummy_verification_ds/targets/ -pd datasets/dummy_verification_ds/predictions/ -bs 2