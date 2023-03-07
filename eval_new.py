#!/usr/bin/python

# Base
import os
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from tqdm.auto import tqdm

import numpy as np

# Torch
import torch
import torchvision
import torch.nn.functional as F
import PIL.Image as Image

import torchmetrics
from torchmetrics.image.inception import InceptionScore
from pytorch_fid import fid_score, inception

from fid import FID
from dataloader import get_data_loader

# Specific Losses like lpips?
import lpips
# from lpipBase import PerceptualLoss ---> Can live without this too ;)

torch.manual_seed(42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

if device == torch.device("cuda"):
    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
else:
    lpips_vgg = lpips.LPIPS(net='vgg')

def evaluator_lpips_loss(y_pred, y):
    loss = lpips_vgg.forward(y_pred, y)
    return loss.mean()

def pil_img(batch):
    arr = []
    for img in batch:
        pil_img = torchvision.transforms.ToPILImage()(img)

        # resize image to 299x299 - for FID & IS
        pil_img = pil_img.resize((299, 299), Image.BICUBIC)
        # make dtype of image from float to torch.uint8
        pil_img = torchvision.transforms.ToTensor()(pil_img)
        pil_img = pil_img.mul(255).byte()      

        arr.append(pil_img)
    return torch.stack(arr)

def run(opts):
    dataloader = get_data_loader(path_pred=f"{opts.predictions_dir}", 
                            path_gt=f"{opts.targets_dir}",
                            batch_size=opts.batch_size)

    # initialize empty torch tensors for each metric
    l1 = torch.tensor([]).to(device)
    ssim_b = []
    iscore = []
    lpips = []
    fid_samples = []

    inception = InceptionScore(device=device)

    for i,  (y_pred, y) in enumerate(tqdm(dataloader)):
        y_pred_2 = pil_img(y_pred)
        y_2 = pil_img(y)

        y_2.save(f"datasets/FID_ref/test_{i}.png")
        y_pred_2.save(f"datasets/FID_pred/test_{i}.png")

        y_pred_2 = y_pred_2.to(device)
        y_2 = y_2.to(device)
        y_pred = y_pred.to(device)
        y = y.to(device)

        # print(evaluator_lpips_loss(y_pred_2, y_2))
        # calculate metrics and keep appending them in the respective tensor
        l1 = torch.cat((l1, F.l1_loss(y_pred, y, reduction='none').mean(dim=(1,2,3))))

        this_ssim = torchmetrics.functional.structural_similarity_index_measure(y_pred, y)
        ssim_b.append(this_ssim)
        this_is = inception.update(y_pred_2)
        iscore.append(this_is)

        lpips.append(evaluator_lpips_loss(y_pred_2, y_2))
        # to compute all at once (FID):
        fid_samples.append([y_pred_2, y_2])

        if i % 100 == 0:
            print(f"L1: {l1[:-1]} | SSIM: {ssim_b[:-1]} | IS: {iscore[:-1]} | LPIPS: {lpips[:-1]}")

    # calculate mean of each metric and obtain the scalar value
    l1_score = l1.mean().item()
    ssim_score = ssim_b.mean().item()
    iscore_score = inception.compute()
    lpips_score = lpips.mean().item()
    psnr_score = 10 * torch.log10(1/l1)

    # Calculate FID
    fid_ans = fid_score.calculate_fid_given_paths(paths=["datasets/FID_ref", "datasets/FID_pred"], 
                                                  batch_size=opts.batch_size, 
                                                  device=device,
                                                  dims=2048)

    # print all the metrics
    print(f"L1: {l1_score}")
    print(f"SSIM: {ssim_score}")
    print(f"FID: {fid_score} | FID ans: {fid_ans}")
    print(f"Inception Score: {iscore_score}")
    print(f"LPIPS: {lpips_score}")
    print(f"PSNR: {psnr_score}")

    # save all the metrics in a text file with experiment name
    with open(f"{opts.exp_name}/metrics.txt", "w") as f:
        f.write(f"L1: {l1_score}")
        f.write(f"SSIM: {ssim_score}")
        f.write(f"FID: {fid_score}")
        f.write(f"Inception Score: {iscore_score}")
        f.write(f"LPIPS: {lpips_score}")
        f.write(f"PSNR: {psnr_score}")


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
    parser.add_argument('--psnr', action='store_true', default=True)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = Path(f"eval_run-{datetime.now().strftime('%Y%m%d_%H_%M_%S')}/")

    assert args.targets_dir is not None, f"[!] Targets' Directory not specified. Usage: ./evaluate.py -td <target_dir>"
    assert args.predictions_dir is not None, f"[!] Predictions' Directory not specified. Usage: ./evaluate.py -pd <preds_dir>"

    run(args)

# ./eval_new.py -en full_test_SPADE -td datasets/full_test/ -pd datasets/synthesized_image/ -bs 16
