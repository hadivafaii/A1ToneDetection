from typing import Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
from .model_utils import print_num_params


class LearnedSwish(nn.Module):
    def __init__(self, slope: float = 1.0):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(float(slope), dtype=torch.float))

    def forward(self, x):
        return self.slope * x * torch.sigmoid(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.signoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.signoid(x)


class NormalizedMSE(nn.Module):
    def __init__(self, mode: str = 'var', dim: int = 0, reduction: str = 'sum'):
        super(NormalizedMSE, self).__init__()
        self.mode = mode
        self.dim = dim
        self.reduction = reduction
        self.fn = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.fn(pred, target)
        if self.mode == 'std':
            loss /= target.std(self.dim, keepdim=True)
        elif self.mode == 'var':
            loss /= target.var(self.dim, keepdim=True)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


class Permute(nn.Module):
    def __init__(self, dims: Tuple[int, ...] = None):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_fn == 'swish':
        return Swish()
    elif activation_fn == 'learned_swish':
        return LearnedSwish(slope=1.0)
    elif activation_fn == 'gelu':
        return nn.GELU()
    else:
        raise RuntimeError("requested activation function '{}' is not supported yet".format(activation_fn))


def get_init_fn(init_range: float = 0.01):
    def init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, mean=0.0, std=init_range)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, val=1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, val=0.0)
    return init_weights


def reparametrize(mu, logsigma):
    std = torch.exp(0.5 * logsigma)
    eps = torch.randn_like(mu).to(mu.device)
    z = mu + std * eps
    return z


def gaussian_residual_kl(delta_mu, log_deltasigma, logsigma):
    """
    :param delta_mu: residual mean
    :param log_deltasigma: log of residual covariance
    :param logsigma: log of prior covariance
    :return: D_KL ( q || p ) where
    q = N ( mu + delta_mu , sigma * deltasigma ), and
    p = N ( mu, sigma )
    """
    return 0.5 * (delta_mu ** 2 / logsigma.exp() + log_deltasigma.exp() - log_deltasigma - 1.0).sum()


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    # computes D_KL ( 1 || 2 )
    return 0.5 * (
            (logsigma1.exp() + (mu1 - mu2) ** 2) / logsigma2.exp() +
            logsigma2 - logsigma1 - 1.0
    ).sum()


def add_wn(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        weight_norm(m)


def add_sn(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        spectral_norm(m)
