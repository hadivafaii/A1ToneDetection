import os
import yaml
import torch
import numpy as np
from torch import nn
from typing import Tuple
from copy import deepcopy as dc
from os.path import join as pjoin
from prettytable import PrettyTable
from .configuration import FeedForwardConfig, VAEConfig
from utils.generic_utils import now


def save_model(
        model: nn.Module,
        comment: str,
        chkpt: int = -1,
):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hash_str = str(hash(frozenset(sorted(to_hash_dict_))))

    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
        type(model).__name__,
        '{}_{}'.format(comment, hash_str),
        '{0:04d}'.format(chkpt),
    )
    os.makedirs(save_dir, exist_ok=True)
    bin_file = pjoin(save_dir, '{:s}.bin'.format(type(model).__name__))
    torch.save(model.state_dict(), bin_file)

    config_file = pjoin(save_dir, '{:s}.yaml'.format(type(model.config).__name__))
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    with open(pjoin(save_dir, '{}.txt'.format(now(exclude_hour_min=False))), 'w') as f:
        f.write("chkpt {:d} saved".format(chkpt))


def load_model(
        keyword: str,
        chkpt_id: int = -1,
        strict: bool = True,
        verbose: bool = False,
        base_dir: str = 'Documents/A1',
):
    match = False
    model_dir = pjoin(os.environ['HOME'], base_dir, 'saved_models')
    for root, dirs, files in os.walk(model_dir):
        match = next(filter(lambda x: keyword in x, dirs), None)
        if match:
            model_dir = pjoin(root, match)
            if verbose:
                print('models found:\nroot: {:s}\nmatch: {:s}'.format(root, match))
            break

    if not match:
        raise RuntimeError('no match found for keyword: {:s}'.format(keyword))

    available_chkpts = sorted(os.listdir(model_dir), key=lambda x: int(x))
    if verbose:
        print('there are {:d} chkpts to load'.format(len(available_chkpts)))
    load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

    if verbose:
        print('\nLoading from:\n{}\n'.format(load_dir))

    config_name = next(filter(lambda s: 'yaml' in s, os.listdir(load_dir)), None)
    with open(pjoin(load_dir, config_name), 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    if 'FeedForwardConfig' in config_name:
        config = FeedForwardConfig(**config_dict)
    elif 'VAEConfig' in config_name:
        config = VAEConfig(**config_dict)
    else:
        raise RuntimeError('unknown config: {}'.format(config_name))

    if type(config).__name__ == 'FeedForwardConfig':
        from .feedforward import TiedAutoEncoder
        loaded_model = TiedAutoEncoder(config, verbose=verbose)
    elif type(config).__name__ == 'VAEConfig':
        raise NotImplementedError
        # from .vae import VAE
        # loaded_model = VAE(config, verbose=verbose)
    else:
        raise RuntimeError("invalid config type encountered")

    bin_file = pjoin(load_dir, '{:s}.bin'.format(type(loaded_model).__name__))
    loaded_model.load_state_dict(torch.load(bin_file), strict=strict)
    loaded_model.eval()

    chkpt = available_chkpts[chkpt_id]
    metadata = {"model_name": str(match), "chkpt": int(chkpt)}

    return loaded_model, metadata


def print_num_params(module: nn.Module):
    t = PrettyTable(['Module Name', 'Num Params'])

    for name, m in module.named_modules():
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if '.' not in name:
            if isinstance(m, type(module)):
                t.add_row(["{}".format(m.__class__.__name__), "{}".format(total_params)])
                t.add_row(['---', '---'])
            else:
                t.add_row([name, "{}".format(total_params)])
    print(t, '\n\n')


def add_weight_decay(model, weight_decay: float = 1e-3, skip_keywords: Tuple[str, ...] = ('bias', 'gain',)):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) <= 1 or any(k in name for k in skip_keywords):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]
    return param_groups


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()

