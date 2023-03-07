import os
from pathlib import Path

import cv2 as cv
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize 

# Enable EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class CustomDataset(Dataset):
    def __init__(self, path_pred: str, path_gt: str, image_transform, label_transform, load2RAM=False):
        self.gt_root = Path(path_gt)
        self.pred_root = Path(path_pred) 

        # Get Paths. DO NOT COMMENT OUT!
        self.data_in_RAM = False
        pred_paths = sorted(list(self.pred_root.glob("*.exr")))
        gt_paths = sorted(list(self.gt_root.glob("*.exr")))
            
        self.data = list(zip(pred_paths, gt_paths))

        # [OPTIONAL] Load images into RAM -- OFF by default
        if load2RAM:
            self.data_in_RAM = True
            import multiprocessing
            from tqdm.auto import tqdm
            pool = multiprocessing.Pool()

            loaderName = "evaluation"
            print("Loading "+loaderName+" predictions and GTs to RAM...")
            preds = tqdm(pool.imap(self.loadSkydome_spade, pred_paths, chunksize=100), total=len(pred_paths))
            gts = tqdm(pool.imap(self.loadSkydome_spade, gt_paths, chunksize=100), total=len(gt_paths))
            self.data = list(zip(preds, gts))
        
        # Transforms
        self.image_transform = image_transform
        self.label_transform = label_transform
        
        
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


def get_data_loader(path_pred: str, path_gt: str, batch_size: int):
    image_transform = Compose([ToTensor()])
    label_transform = Compose([ToTensor()])
    if batch_size:
        dataloader = DataLoader(
            CustomDataset(path_pred=path_pred, path_gt=path_gt, image_transform=image_transform, 
                label_transform=label_transform),
            batch_size=batch_size,
            shuffle=True)
    else: 
        dataloader = None

    return dataloader
