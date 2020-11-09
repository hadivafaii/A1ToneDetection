import os
import yaml
import torch
import numpy as np
from torch import nn
from copy import deepcopy as dc
from os.path import join as pjoin
from prettytable import PrettyTable
from .configuration import FeedForwardConfig
from utils.generic_utils import now


def save_model(model, comment: str, chkpt: int = -1):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hash_str = str(hash(frozenset(sorted(to_hash_dict_))))

    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
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


def load_model(keyword: str, chkpt_id: int = -1, config=None, verbose: bool = False, base_dir: str = 'Documents/A1'):
    _dir = pjoin(os.environ['HOME'], base_dir, 'saved_models')
    available_models = os.listdir(_dir)
    if verbose:
        print('Available models to load:\n{:s}'.format(available_models))

    match_found = False
    model_id = -1
    for i, model_name in enumerate(available_models):
        if keyword in model_name:
            model_id = i
            match_found = True
            break

    if not match_found:
        raise RuntimeError("no match found for keyword: {:s}".format(keyword))

    model_dir = pjoin(_dir, available_models[model_id])
    available_chkpts = sorted(os.listdir(model_dir), key=lambda x: int(x))
    if verbose:
        print('\nAvailable chkpts to load:\n{}'.format(available_chkpts))
    load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

    if verbose:
        print('\nLoading from:\n{}\n'.format(load_dir))

    if config is None:
        config_name = next(filter(lambda s: 'yaml' in s, os.listdir(load_dir)), None)
        with open(pjoin(load_dir, config_name), 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        if 'FeedForwardConfig' in config_name:
            config = FeedForwardConfig(**config_dict)
        elif 'VAEConfig' in config_name:
            raise NotImplementedError
            # config = VAEConfig(**config_dict)

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
    loaded_model.load_state_dict(torch.load(bin_file))
    loaded_model.eval()

    model_name = available_models[model_id]
    chkpt = available_chkpts[chkpt_id]
    metadata = {"model_name": str(model_name), "chkpt": int(chkpt)}

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


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()

