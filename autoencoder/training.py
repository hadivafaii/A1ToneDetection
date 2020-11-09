import os
import numpy as np
from tqdm.notebook import tqdm
from os.path import join as pjoin
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from typing import Union, Tuple
from copy import deepcopy as dc

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .configuration import TrainConfig
from .dataset import create_datasets
from .model_utils import to_np, save_model
from utils.generic_utils import now


class Trainer:
    def __init__(self,
                 model,
                 train_config: TrainConfig,):
        os.environ["SEED"] = str(train_config.random_state)
        torch.manual_seed(train_config.random_state)
        np.random.seed(train_config.random_state)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = model.to(self.device).eval()
        self.config = model.config
        self.train_config = train_config
        self.writer = None

        self.sampler = None
        self.ds_train = None
        self.ds_valid = None
        self._setup_data()

        self.optim = None
        self.optim_schedule = None
        self._setup_optim()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs: Union[int, range], comment):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"

        writer_dir = pjoin(self.train_config.runs_dir, "{}_{}".format(comment, now(exclude_hour_min=True)))
        self.writer = SummaryWriter(writer_dir)

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        pbar = tqdm(epochs_range)
        for epoch in pbar:
            avg_loss = self.iteration(epoch=epoch)
            pbar.set_description('epoch # {:d}, avg loss: {:3f}'.format(epoch, avg_loss))

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                save_model(self.model, comment=comment, chkpt=epoch + 1)

            if (epoch + 1) % self.train_config.eval_freq == 0:
                nb_iters = int(np.ceil(len(self.ds_train) / self.train_config.batch_size))
                global_step = (epoch + 1) * nb_iters
                _ = self.validate(global_step=global_step)

    def iteration(self, epoch=0):
        self.model.train()

        cuml_loss = 0.0
        nb_iters = int(np.ceil(len(self.ds_train) / self.train_config.batch_size))
        pbar = tqdm(range(nb_iters), leave=False)
        for i in pbar:
            global_step = epoch * nb_iters + i

            samples = self.ds_train[list(self.sampler)]
            names, dffs, licks, labels = samples

            loss_list = []
            for name, dff in zip(names, dffs):
                dff = send_to_cuda(dff, self.device)
                y, _ = self.model(name, dff)
                loss_list.append(self.model.criterion(y, dff) / len(dff))
            loss = sum(x for x in loss_list) / len(loss_list)
            _check_for_nans(loss, global_step)
            cuml_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar('extras/lr', self.optim_schedule.get_last_lr()[0], global_step)

        self.optim_schedule.step()
        return cuml_loss / nb_iters

    def validate(self, global_step: int = None):
        self.model.eval()

        samples = self.ds_valid[:]
        names, dffs, licks, labels = samples

        trues = []
        preds = []
        latents = []
        r2s = []
        loss_list = []
        for name, dff in zip(names, dffs):
            dff = send_to_cuda(dff, self.device)
            y, z = self.model(name, dff)
            loss_list.append(self.model.criterion(y, dff) / len(dff))

            dff, y, z = tuple(map(to_np, (dff, y, z)))
            trues.append(dff)
            preds.append(y)
            latents.append(z)
            r2s.append(np.maximum(0.0, r2_score(dff, y, multioutput='raw_values')) * 100)

        loss = sum(x for x in loss_list) / len(loss_list)
        loss = loss.item()
        mean_r2s = [item.mean() for item in r2s]
        mean_r2s = np.mean(mean_r2s)

        if global_step is not None:
            self.writer.add_scalar("loss/valid", loss, global_step)
            self.writer.add_scalar("extras/r2_vld", mean_r2s, global_step)
        else:
            msg = "valid loss: {:.3f},   valid r2 score:  {:.2f} {:s}"
            msg = msg.format(loss, mean_r2s, '%')
            print(msg)

        return names, trues, preds, latents, r2s, labels

    def swap_model(self, new_model):
        self.model = new_model.to(self.device).eval()
        self.config = new_model.config
        self._setup_data()
        self._setup_optim()

    def _setup_data(self):
        sampler, ds_train, ds_valid = create_datasets(self.config, self.train_config)
        self.sampler = sampler
        self.ds_train = ds_train
        self.ds_valid = ds_valid

    def _setup_optim(self):
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,)
        self.optim_schedule = CosineAnnealingLR(
            self.optim,
            T_max=self.train_config.scheduler_period,
            eta_min=self.train_config.eta_min,)


def send_to_cuda(x, device, dtype=torch.float32) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if isinstance(x, (tuple, list)):
        return tuple(map(lambda z: z.to(device=device, dtype=dtype, non_blocking=False), x))
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=False)
    else:
        return torch.tensor(x, device=device, dtype=dtype)


def _check_for_nans(loss, global_step):
    if torch.isnan(loss).sum().item():
        msg = "WARNING: nan encountered in loss. optimizer will detect this and skip. step = {}"
        msg = msg.format(global_step)
        raise RuntimeWarning(msg)
