import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor


# Core Noise Prediction Losses

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def charbonnier_loss(pred, target, epsilon=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + epsilon ** 2))


# SSIM Loss


def _gaussian(window_size, sigma):
    center = window_size // 2
    x_vals = torch.arange(window_size).float()
    gauss = torch.exp(-((x_vals - center) ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_loss(pred, target, window_size=11, size_average=True):
    channel = pred.size(1)
    window = _create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return 1 - ssim_map.mean() if size_average else 1 - ssim_map


# VGG Perceptual Loss (optional)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=['features.3', 'features.8']):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval()
        self.extractor = create_feature_extractor(vgg, return_nodes={k: k for k in layers})
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_feat = self.extractor(pred)
        target_feat = self.extractor(target)
        loss = 0.0
        for layer in pred_feat:
            loss += F.l1_loss(pred_feat[layer], target_feat[layer])
        return loss

# Hybrid Loss (MSE + SSIM + KL)

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        ssim = ssim_loss(pred, target)

        mu1, std1 = pred.mean(), pred.std()
        mu2, std2 = target.mean(), target.std()
        kl = torch.log(std2 / (std1 + 1e-8) + 1e-8) + \
             (std1 ** 2 + (mu1 - mu2) ** 2) / (2 * (std2 ** 2 + 1e-8)) - 0.5

        return self.alpha * mse + self.beta * ssim + self.gamma * kl


# Dynamic loss function selector

def get_loss_fn(name, alpha=1.0, beta=0.1, gamma=0.01):
    name = name.lower()
    if name == "mse":
        return mse_loss
    elif name == "l1":
        return l1_loss
    elif name == "charbonnier":
        return charbonnier_loss
    elif name == "ssim":
        return ssim_loss
    elif name == "perceptual":
        return VGGPerceptualLoss()
    elif name == "hybrid_light":
        return HybridLoss(alpha, beta, gamma)
    else:
        raise ValueError(f"Unknown loss function: {name}")
