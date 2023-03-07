import os
import utils
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import PIL.Image as Image
from fid import FID
from copy import deepcopy

debug = False

class SingleDataset(Dataset):
    def __init__(self, skydb_path, raw_path, csv_path, res=128, dataset_bin='all', hw=False, skygan=False, skynet=False, is_lm=False, prague=False, expose=False):
        self.samples = []

        cloudiness_min = 3285
        cloudiness_max = 525266
        bin_thresholds = {}
        bin_thresholds['su'] = (cloudiness_max - cloudiness_min) * (1 * 0.125) + cloudiness_min
        bin_thresholds['ms'] = (cloudiness_max - cloudiness_min) * (3 * 0.125) + cloudiness_min
        bin_thresholds['pc'] = (cloudiness_max - cloudiness_min) * (5 * 0.125) + cloudiness_min
        bin_thresholds['mc'] = (cloudiness_max - cloudiness_min) * (7 * 0.125) + cloudiness_min
        bin_thresholds['oc'] = cloudiness_max

        annotations = utils.load_annotations(csv_path)
        s2l_map = utils.get_skyangular2latlong_map(res)
        l2s_map = utils.get_latlong2skyangular_map(res)

        if debug: annotations = annotations[:10]

        for annotation in tqdm(annotations, desc='Loading dataset', ncols=80):
            tag = annotation['filename'].replace('data/', '')[:-1].split('/')

            cur_bin = 'none'
            if int(np.rad2deg(annotation['zenith']) % 180) >= 70:
                cur_bin = 'ss'
            else:
                for bin in bin_thresholds:
                    if annotation['cloudiness'] <= bin_thresholds[bin]:
                        cur_bin = bin
                        break

            if cur_bin != dataset_bin and dataset_bin != 'all': continue

            if not skygan:
                lm = utils.load_exr(os.path.join(skydb_path, annotation['filename'], 'lm_' + str(128) + '.exr'), (128, 128))
            else:
                lm = utils.load_exr(os.path.join(raw_path.replace('skygan', 'hw'), tag[0] + '_' + tag[1] + '_hw.exr'), (res, res))

            gt = utils.load_exr(os.path.join(skydb_path, annotation['filename'], 'gt_' + str(128) + '.exr'), (128, 128))

            pred_path = os.path.join(raw_path, tag[0] + '_' + tag[1] + '.exr')
            if hw:
                pred_path = pred_path.replace('.exr', '_hw.exr')
            if is_lm:
                pred = deepcopy(lm)
            else:
                pred = utils.load_exr(pred_path, (res, res))

            if skynet:
                pred = utils.fix_skynet(pred, tag[0] + '_' + tag[1],
                                        annotations, s2l_map, l2s_map, res)

            if prague:
                pred = pred ** (1 / 2.2)
            if expose:
                pred = utils.expose_exr(pred)
            if skygan:
                lm = utils.expose_exr(lm)

            if res > 128:
                pred = utils.resize_skyangular(pred, res, 128, numpy=True)


            #utils.save_ldr(os.path.join(raw_path, 'gt.jpg'), utils.hdr2ldr_envmap(gt))
            #utils.save_ldr(os.path.join(raw_path, 'lm.jpg'), utils.hdr2ldr_envmap(lm))
            #utils.save_ldr(os.path.join(raw_path, 'pred.jpg'), utils.hdr2ldr_envmap(pred))

            lm = torch.tensor(np.moveaxis(lm, -1, 0)).float()
            gt = torch.tensor(np.moveaxis(gt, -1, 0)).float()
            pred = torch.tensor(np.moveaxis(pred, -1, 0)).float()

            self.samples.append({'pred': pred, 'lm': lm, 'gt': gt, 'filename': annotation['filename']})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def calculate_RMSE(src, target, scale_invariant=False):
    src = src.flatten(start_dim=1)
    target = target.flatten(start_dim=1)

    if scale_invariant:
        for idx, _ in enumerate(src):
            alpha = torch.dot(target[idx], src[idx])
            alpha /= torch.dot(src[idx], src[idx])
            src[idx] *= alpha

    error = (target - src)**2
    error = error.mean(dim=1)
    error = torch.sqrt(error + np.finfo(np.float32).eps)
    return error


def resize_fid(batch, size):
    arr = []
    for img in batch:
        pil_img = torchvision.transforms.ToPILImage()(img)
        resized_img = pil_img.resize(size, Image.Resampling.BILINEAR)
        arr.append(torchvision.transforms.ToTensor()(resized_img))
    return torch.stack(arr)



use_bins = True
nametag = 'prague'
res = 256
# lm 128, gt 128,

raw_path = os.path.join(os.path.abspath('..'), 'raw', nametag)
out_path = os.path.join(os.path.abspath('..'), 'out', 'statistics', 'gpgpu')
skydb_path = '/home/lvrma/dataset/SkyAngularDB'
csv_path = os.path.join(skydb_path, 'test_annotations.csv')
ltm_path =  os.path.join(skydb_path, 'ltm_sphere_128.npz')

hw = True if 'hw' == nametag else False
skygan = True if 'skygan' == nametag else False
prague = True if 'prague' == nametag else False
skynet = True if 'skynet' == nametag else False
is_lm = True if 'lm' == nametag else False
device = 'cuda:0'

expose = True if nametag in ['hw', 'skygan', 'skynet', 'prague'] else False

if use_bins:
    bins = ['su', 'ms', 'pc', 'mc', 'oc', 'ss', 'all']
else:
    bins = ['all']

for bin in bins:
    print('---------------------------------- ' + bin + ' ------------------------------')
    dataset = SingleDataset(skydb_path, raw_path, csv_path, res, bin, hw=hw, skygan=skygan, skynet=skynet, is_lm=is_lm, prague=prague, expose=expose)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    fid_net = FID(device)

    rmse = torch.tensor([])
    si_rmse = torch.tensor([])
    ltm_gt_rmse = torch.tensor([])
    ltm_gt_si_rmse = torch.tensor([])
    ltm_lm_rmse = torch.tensor([])
    ltm_lm_si_rmse = torch.tensor([])
    fid_samples = []

    loop = tqdm(loader, total=len(loader), leave=True, ncols=80, colour='green', desc=f'Benchmarking')
    for idx, sample in enumerate(loop):
        pred = sample['pred']
        gt = sample['gt']

        rmse = torch.cat((rmse, calculate_RMSE(deepcopy(pred.cpu()), deepcopy(gt.cpu()))), dim=0)
        si_rmse = torch.cat((si_rmse, calculate_RMSE(deepcopy(pred.cpu()), deepcopy(gt.cpu()), scale_invariant=True).cpu()), dim=0)

        ltm_lm_rmse = torch.cat((ltm_lm_rmse, calculate_RMSE(render_pred, render_lm).cpu()), dim=0)
        ltm_lm_si_rmse = torch.cat((ltm_lm_si_rmse, calculate_RMSE(render_pred, render_lm, scale_invariant=True).cpu()), dim=0)

        ltm_gt_rmse = torch.cat((ltm_gt_rmse, calculate_RMSE(render_pred, render_gt).cpu()), dim=0)
        ltm_gt_si_rmse = torch.cat((ltm_gt_si_rmse, calculate_RMSE(render_pred, render_gt, scale_invariant=True).cpu()), dim=0)

        fid_pred = resize_fid((deepcopy(pred.cpu())**(1/2.2)).clamp(0, 1), (299, 299)).float()
        fid_gt = resize_fid((deepcopy(gt.cpu())**(1/2.2)).clamp(0, 1), (299, 299)).float()
        fid_samples.append((fid_pred, fid_gt))

        if idx == 0 and bin == bins[0]:
            for s in range(1):
                utils.save_ldr(os.path.join(raw_path, str(s) + '_gt.jpg'), utils.hdr2ldr_envmap(np.moveaxis(gt[s].cpu().numpy(), 0, -1)))
                utils.save_ldr(os.path.join(raw_path, str(s) + '_lm.jpg'), utils.hdr2ldr_envmap(np.moveaxis(lm[s].cpu().numpy(), 0, -1)))
                utils.save_ldr(os.path.join(raw_path, str(s) + '_pred.jpg'), utils.hdr2ldr_envmap(np.moveaxis(pred[s].cpu().numpy(), 0, -1)))
                utils.save_ldr(os.path.join(raw_path, str(idx) + '_' + str(s) + '_ltm_gt.jpg'), utils.expose_exr(np.moveaxis(render_gt[s].cpu().numpy() / 100, 0, -1), value=100))
                utils.save_ldr(os.path.join(raw_path, str(idx) + '_' + str(s) + '_ltm_lm.jpg'), utils.expose_exr(np.moveaxis(render_lm[s].cpu().numpy() / 100, 0, -1), value=100))
                utils.save_ldr(os.path.join(raw_path, str(idx) + '_' + str(s) + '_ltm_pred.jpg'), utils.expose_exr(np.moveaxis(render_pred[s].cpu().numpy() / 100, 0, -1), value=100))
                utils.save_ldr(os.path.join(raw_path, str(s) + '_fid_gt.jpg'), utils.hdr2ldr_envmap(np.moveaxis(fid_pred[s].cpu().numpy(), 0, -1)))
                utils.save_ldr(os.path.join(raw_path, str(s) + '_fid_lm.jpg'), utils.hdr2ldr_envmap(np.moveaxis(fid_gt[s].cpu().numpy(), 0, -1)))


    rmse_score = rmse.mean().item()
    si_rmse_score = si_rmse.mean().item()
    ltm_gt_rmse_score = ltm_gt_rmse.mean().item()
    ltm_gt_si_rmse_score = ltm_gt_si_rmse.mean().item()
    ltm_lm_rmse_score = ltm_lm_rmse.mean().item()
    ltm_lm_si_rmse_score = ltm_lm_si_rmse.mean().item()


    if len(fid_samples) > 0:
        for sample in tqdm(fid_samples, desc='FID', ncols=80):
            fid_net.update(sample[0], sample[1])

        fid_score = fid_net.compute()
        fid_net.reset()
    else:
        fid_score = 0

    print('FID: ' + str(round(fid_score, 2)))
    print('RMSE: ' + str(round(rmse_score, 2)))
    print('si-RMSE: ' + str(round(si_rmse_score, 2)))
    print('GT LTM RMSE: ' + str(round(ltm_gt_rmse_score, 2)))
    print('GT LTM si-RMSE: ' + str(round(ltm_gt_si_rmse_score, 2)))
    print('In LTM RMSE: ' + str(round(ltm_lm_rmse_score, 2)))
    print('In LTM si-RMSE: ' + str(round(ltm_lm_si_rmse_score, 2)))
    print(str(len(rmse)) + ' samples')

    np.savez(os.path.join(out_path, 'benchmark_rerun_' + nametag + '_' + bin + '.npz'),
             fid_score=fid_score,
             rmse_score=rmse_score,
             si_rmse_score=si_rmse_score,
             ltm_gt_rmse_score=ltm_gt_rmse_score,
             ltm_gt_si_rmse_score=ltm_gt_si_rmse_score,
             ltm_lm_rmse_score=ltm_lm_rmse_score,
             ltm_lm_si_rmse_score=ltm_lm_si_rmse_score,
             rmse=rmse.numpy(),
             si_rmse=si_rmse.numpy(),
             ltm_gt_rmse=ltm_gt_rmse.numpy(),
             ltm_gt_si_rmse=ltm_gt_si_rmse.numpy(),
             ltm_lm_rmse=ltm_lm_rmse.numpy(),
             ltm_lm_si_rmse=ltm_lm_si_rmse.numpy())
