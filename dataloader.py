import os
from pathlib import Path

import cv2 as cv
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

import numpy as np
import cv2 as cv

# Enable EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class CustomDataset(Dataset):
    def __init__(self, path_pred: str, path_gt: str, image_transform, load2RAM=False, max_samples=None):
        self.gt_root = Path(path_gt)
        self.pred_root = Path(path_pred) 

        # Get Paths. DO NOT COMMENT OUT!
        pred_paths = list(self.pred_root.glob("*.exr"))
        gt_paths = list(self.gt_root.glob("*.exr"))

        if max_samples is not None and len(pred_paths) > max_samples:
            pred_paths = pred_paths[:max_samples]
            gt_paths = gt_paths[:max_samples]

        # [OPTIONAL] Load images into RAM -- OFF by default
        self.data_in_RAM = load2RAM
        if load2RAM:
            import multiprocessing
            from tqdm.auto import tqdm
            pool = multiprocessing.Pool()

            loaderName = "evaluation"
            print("Loading "+loaderName+" predictions and GTs to RAM...")
            preds = tqdm(pool.imap(self.loadSkydome_spade, pred_paths, chunksize=100), total=len(pred_paths))
            gts = tqdm(pool.imap(self.loadSkydome_spade, gt_paths, chunksize=100), total=len(gt_paths))
            self.data = list(zip(preds, gts))
        else:
            self.data = list(zip(pred_paths, gt_paths))
        
        # Transforms
        self.label_transform = image_transform
        self.image_transform = image_transform
        
    def loadSkydome_spade(self, p):
        img = cv.imread(p, (cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = cv.resize(img, (256,256), interpolation=cv.INTER_CUBIC)
        return img

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data_in_RAM:
            pred, gt = self.data[idx]
        else: 
            pred_path, gt_path = self.data[idx]
            pred = self.loadSkydome_spade(pred_path.as_posix())
            gt = self.loadSkydome_spade(gt_path.as_posix())

        pred = self.image_transform(pred) # ToTensor
        gt = self.label_transform(gt) # ToTensor

        return pred, gt


def get_data_loader(path_pred: str, path_gt: str, batch_size: int, max_samples=None):
    tensor_transform = Compose([ToTensor()])

    dataloader = DataLoader(
        CustomDataset(path_pred=path_pred, path_gt=path_gt, image_transform=tensor_transform, max_samples=max_samples),
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader
