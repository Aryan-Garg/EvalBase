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
from ignite.metrics import Loss, SSIM, FID, MeanAbsoluteError, InceptionScore, PSNR
from ignite.utils import manual_seed
from ignite.handlers import Timer

from dataloader import get_data_loader


# Setup seed to have same model's initialization:
manual_seed(31415)

# Hardware
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Evaluating on device: {}".format(device))


import lpips
lpips_vgg = None

def process_function(engine, batch):
    # Get batch
    y_pred, y = batch
    y_pred = y_pred.to(device)
    y = y.to(device)
    return y_pred, y

def evaluator_loss_lpips(y_pred, y):
    loss = lpips_vgg.forward(y_pred, y)
    return loss.mean()

def formatter_IS(output):
    y_pred, y = output
    return y_pred  # output format is according to `Accuracy` docs

def setup_metrics(opts):
    # Define & Initializer metrics in dictionary
    metrics=dict()

    # Default (Sanity) Metric
    metrics["PSNR"] = PSNR(data_range=1.0, device=device)

    # ### TODO: Add more metrics here
    if opts.fid:
        metrics["FID"] = FID(device=device)
    if opts.ssim:
        # For tmed output (HDR)
        metrics["SSIM"] = SSIM(data_range=1.0, device=device)
    if opts.l1:
        metrics["L1"] = MeanAbsoluteError(device=device)
    if opts.iscore:
        metrics["IS"] = InceptionScore(output_transform=formatter_IS,device=device)
    if opts.lpips:
        global lpips_vgg
        lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        metrics["lpips"] = Loss(evaluator_loss_lpips,device=device)

    return metrics


def save_eval_results(RUN_NAME: str, metrics: dict):
    with open(f"results/{RUN_NAME}.txt", "a") as f:
        f.write("Evaluation Metrics:\n")
        for k,v in metrics.items():
            f.write(f"{k} = {v}\n")


def run(opts):
    engine = Engine(process_function)
    metrics = setup_metrics(opts)
    for k in metrics:
        metrics[k].attach(engine, k)


    dataloader = get_data_loader(path_pred=f"{opts.predictions_dir}", 
                            path_gt=f"{opts.targets_dir}",
                            batch_size=opts.batch_size,
                            max_samples=100 if opts.testing else None)

    # Run evaluation
    state = engine.run(dataloader)
    for k in metrics:
        print(f"{k}: {state.metrics[k]}")

    # Save results
    save_eval_results(opts.exp_name, metrics)


if __name__ == "__main__":
    parser = ArgumentParser(prog = 'Evaluation Metrics',
                    description = 'Runs Evaluation Metrics on results. (Stand-alone evaluator)',
                    epilog = 'Rights reserved by professor Jean-Francois Lalonde')
    parser.add_argument('-en','--exp_name', type=str, default=None, help="Name of experiment. Used for saving results.")
    parser.add_argument('-T', '--testing', action='store_true', default=False, help="Caps the number of samples to 100 for testing purposes")
    parser.add_argument('-td', '--targets_dir', type=str, default="data/GT", help="GT directory.")
    parser.add_argument('-pd', '--predictions_dir', type=str, default="data/PRED", help="Predictions' dir.")
    parser.add_argument('-bs', '--batch_size', type=int, default=15, help="Batch Size.")
    
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

# ./evaluate.py -en full_test_SPADE -td datasets/full_pysolar_test/full_gt_test/ -pd datasets/full_pysolar_test/synthesized_image/ -bs 2