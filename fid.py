import torch
import torch.nn as nn
import torchvision

class FID(nn.Module):
    def __init__(self, device='cpu'):
        super(FID, self).__init__()

        self.device = device
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1

        self.model = torchvision.models.inception_v3(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()

        self.x_src = None
        self.x_target = None
        self.xx_src = None
        self.xx_target = None
        self.count_src = 0
        self.count_target = 0
        self.reset()


    def update(self, data_src, data_target):
        with torch.no_grad():
            data_src = (2 * data_src - 1).to(self.device)
            data_target = (2 * data_target - 1).to(self.device)

            for sample_src, sample_target in zip(data_src, data_target):
                sample_src = torch.unsqueeze(sample_src, dim=0)
                sample_target = torch.unsqueeze(sample_target, dim=0)

                self.count_src += len(sample_src)
                self.count_target += len(sample_target)

                pred = self.model(sample_src)
                self.x_src += pred.sum(dim=0, keepdim=True).T
                self.xx_src += (pred[:, None] * pred[..., None]).sum(dim=0)

                pred = self.model(sample_target)
                self.x_target += pred.sum(dim=0, keepdim=True).T
                self.xx_target += (pred[:, None] * pred[..., None]).sum(dim=0)


    def reset(self):
        self.x_src = torch.zeros((2048, 1)).to(self.device)
        self.xx_src = torch.zeros((2048, 2048)).to(self.device)
        self.count_src = 0

        self.x_target = torch.zeros((2048, 1)).to(self.device)
        self.xx_target = torch.zeros((2048, 2048)).to(self.device)
        self.count_target = 0


    def compute(self):
        mean_src = self.x_src / self.count_src
        cov_src = self.xx_src / self.count_src - mean_src @ mean_src.T

        mean_target = self.x_target / self.count_target
        cov_target = self.xx_target / self.count_target - mean_target @ mean_target.T

        eigs = torch.linalg.eigvals(cov_src @ cov_target)
        means = ((mean_src - mean_target)**2).sum()
        covs = cov_src.trace() + cov_target.trace()
        fid = means + covs - 2 * eigs.real.clamp(min=0).sqrt().sum()

        return fid.item()