import os
import numpy as np
from tqdm.notebook import tqdm
from os.path import join as pjoin
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from typing import Union, Tuple
from sklearn.metrics import accuracy_score
from copy import deepcopy as dc

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .configuration import TrainConfig
from .feedforward import TiedAutoEncoder, Classifier
from .dataset import create_a1_dataset, create_clf_dataset
from .model_utils import to_np, save_model
from utils.generic_utils import now


class BaseTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 train_config: TrainConfig,
                 **kwargs,):
        super(BaseTrainer, self).__init__()
        kwargs_defaults = {
            'verbose': False,
            'source_trainer': None,
        }
        for k, v in kwargs_defaults.items():
            if k not in kwargs:
                kwargs[k] = v

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
        self.dl_train = None
        self.dl_valid = None
        self.setup_data(**kwargs)

        self.optim = None
        self.optim_schedule = None
        self.setup_optim()

        if kwargs['verbose']:
            print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs: Union[int, range], comment):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"

        writer_dir = pjoin(
            self.train_config.runs_dir,
            type(self.model).__name__,
            "{}".format(now(exclude_hour_min=True)),
            "{}".format(comment),
        )
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
                _ = self.validate(global_step, verbose=False)

    def iteration(self, epoch: int = 0):
        raise NotImplementedError

    def validate(self, global_step: int = None, verbose: int = True):
        raise NotImplementedError

    def extract_features(self):
        pass

    def xtract(self, mode: str):
        pass

    def setup_data(self, **kwargs):
        raise NotImplementedError

    def swap_model(self, new_model, full_setup: bool = True):
        self.model = new_model.to(self.device).eval()
        self.config = new_model.config
        if full_setup:
            self.setup_data()
            self.setup_optim()

    def setup_optim(self):
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,)
        self.optim_schedule = CosineAnnealingLR(
            self.optim,
            T_max=self.train_config.scheduler_period,
            eta_min=self.train_config.eta_min,)

    def to_cuda(self, x, dtype=torch.float32) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(x, (tuple, list)):
            return tuple(map(lambda z: torch.tensor(z, device=self.device, dtype=dtype), x))
        else:
            return torch.tensor(x, device=self.device, dtype=dtype)


class AETrainer(BaseTrainer):
    def __init__(self,
                 model: TiedAutoEncoder,
                 train_config: TrainConfig,
                 **kwargs,):
        super(AETrainer, self).__init__(model, train_config, **kwargs)

    def iteration(self, epoch: int = 0):
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
                dff = self.to_cuda(dff)
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
        avg_loss = cuml_loss / nb_iters
        return avg_loss

    def validate(self, global_step: int = None, verbose: bool = True):
        self.model.eval()

        samples = self.ds_valid[:]
        names, dffs, licks, labels = samples

        trues = []
        preds = []
        latents = []
        r2s = []
        loss_list = []
        for name, dff in zip(names, dffs):
            dff = self.to_cuda(dff)
            with torch.no_grad():
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
        if verbose:
            msg = "valid loss: {:.3f},   valid r2 score:  {:.2f} {:s}"
            msg = msg.format(loss, mean_r2s, '%')
            print(msg)

        output = {
            'names': names,
            'trues': trues,
            'preds': preds,
            'latents': latents,
            'labels': labels,
            'r2s': r2s,
            'loss': loss,
        }
        return output

    def extract_features(self):
        output_train = self.xtract('train')
        output_valid = self.xtract('valid')

        return output_train, output_valid

    def xtract(self, mode: str):
        self.model.eval()

        if mode == 'train':
            samples = self.ds_train[:]
        elif mode == 'valid':
            samples = self.ds_valid[:]
        else:
            raise NotImplementedError("invalid mode: {}".format(mode))

        names, dffs, licks, labels = samples
        pbar = tqdm(zip(names, dffs, labels), total=len(names), desc="xtracting {}".format(mode), leave=False)
        x_list = []
        z_list = []
        for name, dff, lbl in pbar:
            dff = self.to_cuda(dff)
            with torch.no_grad():
                x = self.model.embedding(name, dff)
                z = self.model(name, dff)[1]
            x_list.append(np.expand_dims(to_np(x), axis=0))
            z_list.append(np.expand_dims(to_np(z), axis=0))
        x = np.concatenate(x_list)
        z = np.concatenate(z_list)
        y = np.array([self.model.config.l2i[lbl] for lbl in labels])

        output = {
            'names': names,
            'labels': labels,
            'x': x,
            'y': y,
            'z': z,
        }
        return output

    def setup_data(self, **kwargs):
        sampler, ds_train, ds_valid = create_a1_dataset(self.config, self.train_config)
        self.sampler = sampler
        self.ds_train = ds_train
        self.ds_valid = ds_valid


class CLFTrainer(BaseTrainer):
    def __init__(self,
                 model: Classifier,
                 train_config: TrainConfig,
                 **kwargs,):
        super(CLFTrainer, self).__init__(model, train_config, **kwargs)

    def iteration(self, epoch: int = 0):
        self.model.train()

        cuml_loss = 0.0
        nb_iters = len(self.dl_train)
        for i, (x, y) in enumerate(self.dl_train):
            global_step = epoch * nb_iters + i

            x = self.to_cuda(x)
            y_pred = self.model(x)
            y = self.to_cuda(y, dtype=torch.long)
            loss = self.model.criterion(y_pred, y) / self.dl_train.batch_size
            _check_for_nans(loss, global_step)
            cuml_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar('extras/lr', self.optim_schedule.get_last_lr()[0], global_step)

        self.optim_schedule.step()
        avg_loss = cuml_loss / nb_iters
        return avg_loss

    def validate(self, global_step: int = None, verbose: bool = True):
        self.model.eval()

        preds = []
        loss_list = []
        for (x, y) in self.dl_valid:
            x = self.to_cuda(x)
            with torch.no_grad():
                y_pred = self.model(x)
            preds.append(to_np(y_pred))
            y = self.to_cuda(y, dtype=torch.long)
            loss = self.model.criterion(y_pred, y) / self.dl_valid.batch_size
            loss_list.append(loss.item())
        loss = np.mean(loss_list)

        y_pred = np.concatenate(preds, axis=0)
        preds = np.argmax(y_pred, axis=-1)
        trues = self.dl_valid.dataset.y
        accuracy = accuracy_score(trues, preds) * 100

        if global_step is not None:
            self.writer.add_scalar("loss/valid", loss, global_step)
            self.writer.add_scalar("extras/accuracy", accuracy, global_step)
        if verbose:
            msg = "valid loss: {:.3f},   valid accuracy score:  {:.2f} {:s}"
            msg = msg.format(loss, accuracy, '%')
            print(msg)

        output = {
            'trues': trues,
            'preds': preds,
            'accuracy': accuracy,
            'loss': loss,
        }
        return output

    def setup_data(self, **kwargs):
        ae_trainer = kwargs['source_trainer']
        output_train, output_valid = ae_trainer.extract_features()
        dl_train, dl_valid = create_clf_dataset(output_train, output_valid, self.train_config.batch_size)
        self.dl_train = dl_train
        self.dl_valid = dl_valid
        self.ds_train = dl_train.dataset
        self.ds_valid = dl_valid.dataset


def _check_for_nans(loss, global_step: int):
    if torch.isnan(loss).sum().item():
        msg = "nan encountered in loss. optimizer will detect this and skip. global step = {}"
        msg = msg.format(global_step)
        raise RuntimeWarning(msg)
