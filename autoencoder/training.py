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
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

from .vae import VAE
from .configuration import TrainConfig
from .feedforward import TiedAutoEncoder, Classifier
from .dataset import create_a1_dataset, create_clf_dataset
from .model_utils import to_np, save_model, add_weight_decay
from utils.generic_utils import now


class BaseTrainer(object):
    def __init__(self, model: nn.Module, train_config: TrainConfig, **kwargs):
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
            if self.optim_schedule is not None:
                self.optim_schedule.step()

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
        params = add_weight_decay(self.model, self.train_config.weight_decay)
        self.optim = AdamW(
            params=params,
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
            betas=(self.train_config.beta1, self.train_config.beta2),
        )

        if self.train_config.scheduler_type == 'cosine':
            self.optim_schedule = CosineAnnealingLR(
                optimizer=self.optim,
                T_max=self.train_config.scheduler_period,
                eta_min=self.train_config.eta_min,
            )
        elif self.train_config.scheduler_type == 'exponential':
            self.optim_schedule = ExponentialLR(
                optimizer=self.optim,
                gamma=self.train_config.scheduler_gamma,
            )
        elif self.train_config.scheduler_type == 'step':
            self.optim_schedule = StepLR(
                optimizer=self.optim,
                step_size=self.train_config.scheduler_period,
                gamma=self.train_config.scheduler_gamma,
            )

    def inspect_grad_norm(self, global_step: int):
        # total_norm = 0.0
        # for p in filter(lambda x: x.grad is not None, self.model.parameters()):
        #     total_norm += p.grad.data.norm(2).item() ** 2
        # total_norm = np.sqrt(total_norm)

        norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_config.grad_clip).item()
        self.writer.add_scalar("extras/grad_norm", norm, global_step)

        if norm > self.train_config.skip_threshold:
            print("global step = {} ### gradient norm {:.1f} too large. skipping".format(global_step, norm))
            return True
        else:
            return False

    def to_cuda(self, x, dtype=torch.float32) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if isinstance(x, (tuple, list)):
            if all(torch.is_tensor(item) for item in x):
                return tuple(map(lambda z: z.to(device=self.device, dtype=dtype), x))
            else:
                return tuple(map(lambda z: torch.tensor(z, device=self.device, dtype=dtype), x))
        else:
            if torch.is_tensor(x):
                return x.to(device=self.device, dtype=dtype)
            else:
                return torch.tensor(x, device=self.device, dtype=dtype)


class VAETrainer(BaseTrainer):
    def __init__(self, model: VAE, train_config: TrainConfig, **kwargs):
        super(VAETrainer, self).__init__(model, train_config, **kwargs)

    def iteration(self, epoch: int = 0):
        self.model.train()

        cuml_loss = 0.0
        nb_iters = int(np.ceil(len(self.ds_train) / self.train_config.batch_size))
        pbar = tqdm(range(nb_iters), leave=False)
        for i in pbar:
            global_step = epoch * nb_iters + i

            samples = self.ds_train[list(self.sampler)]
            dff = self.to_cuda(samples['dff'], dtype=torch.float)
            lick = self.to_cuda(samples['lick'], dtype=torch.float)
            name, label, freq = self.to_cuda(
                [samples['name'], samples['label'], samples['freq']], dtype=torch.long)
            inputs_dict = dict(name=name, label=label, freq=freq, dff=dff, lick=lick)

            outputs_dict, predictions_dict = self.model(inputs_dict)
            loss, loss_dict = self.model.compute_loss(
                inputs_dict=inputs_dict,
                outputs_dict=outputs_dict,
                predictions_dict=predictions_dict,
                coeffs=self.train_config.loss_coeffs,
            )
            if inspect_loss(loss, global_step):
                continue

            self.optim.zero_grad()
            loss.backward()
            if self.inspect_grad_norm(global_step):
                continue
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
                self.writer.add_scalar("loss/dff",  loss_dict['dff'].item(), global_step)
                self.writer.add_scalar("loss/lick", loss_dict['lick'].item(), global_step)
                for level in range(self.model.config.nb_levels):
                    self.writer.add_scalar("loss/kl_{:d}".format(level), loss_dict['kl'][level].item(), global_step)
                _mean_kl = sum(x for x in loss_dict['kl']).item() / self.model.config.nb_levels
                self.writer.add_scalar("loss/kl_mean", _mean_kl, global_step)

            self.writer.add_scalar('extras/lr', self.optim_schedule.get_last_lr()[0], global_step)
            cuml_loss += loss.item()

        return cuml_loss / nb_iters

    def validate(self, global_step: int = None, verbose: bool = True):
        self.model.eval()

        samples = self.ds_valid[:]
        dff = self.to_cuda(samples['dff'], dtype=torch.float)
        lick = self.to_cuda(samples['lick'], dtype=torch.float)
        name, label, freq = self.to_cuda(
            [samples['name'], samples['label'], samples['freq']], dtype=torch.long)
        inputs_dict = dict(name=name, label=label, freq=freq, dff=dff, lick=lick)

        # TODO: this might need to be batched as well
        with torch.no_grad():
            outputs_dict, predictions_dict = self.model(inputs_dict)
            loss, loss_dict = self.model.compute_loss(
                inputs_dict=inputs_dict,
                outputs_dict=outputs_dict,
                predictions_dict=predictions_dict,
                coeffs=self.train_config.loss_coeffs,
            )

        # TODO: add a to_np step to everything
        pred_dffs = list(map(to_np, predictions_dict['dff']))
        pred_lick = to_np(predictions_dict['lick'])

        accuracy_l = inputs_dict['label'].eq(predictions_dict['indxs_l']).float().mean().item() * 100
        accuracy_f = inputs_dict['freq'].eq(predictions_dict['indxs_f']).float().mean().item() * 100
        accuracy_n = inputs_dict['name'].eq(predictions_dict['indxs_n']).float().mean().item() * 100

        # Rest is from previous
        dff_r2s = []
        for pred, true in zip(pred_dffs, samples['dff']):
            dff_r2s.append(np.maximum(0.0, r2_score(true, pred, multioutput='raw_values')) * 100)

        mean_dff_r2s = [item.mean() for item in dff_r2s]
        mean_dff_r2s = np.mean(mean_dff_r2s)

        if global_step is not None:
            self.writer.add_scalar("loss/valid", loss, global_step)
            self.writer.add_scalar("extras/accuracy_labels", accuracy_l, global_step)
            self.writer.add_scalar("extras/accuracy_freqs", accuracy_f, global_step)
            self.writer.add_scalar("extras/accuracy_names", accuracy_n, global_step)
            self.writer.add_scalar("extras/dff_r2", mean_dff_r2s, global_step)
        if verbose:
            msg = "valid loss: {:.3f},   valid r2 score:  {:.2f} {:s}\n"
            msg += "accuracy scores   ---   labels :  {:.2f} {:s},  freqs :  {:.2f} {:s},  names :  {:.2f} {:s}"
            msg = msg.format(loss, mean_dff_r2s, '%', accuracy_l, '%', accuracy_f, '%', accuracy_n, '%')
            print(msg)

        output = {
            'dff': pred_dffs,
            'lick': pred_lick,
            'dff_r2': dff_r2s,
            'loss': loss,
        }
        return output

    # TODO: fix xtract operation for VAE (much later)
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
            if inspect_loss(loss, global_step):
                pass
            cuml_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar('extras/lr', self.optim_schedule.get_last_lr()[0], global_step)
        return cuml_loss / nb_iters

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
        for i, (x, y, z) in enumerate(self.dl_train):
            global_step = epoch * nb_iters + i

            x = self.to_cuda(x)
            y_pred = self.model(x)
            y = self.to_cuda(y, dtype=torch.long)
            loss = self.model.criterion(y_pred, y) / self.dl_train.batch_size
            if inspect_loss(loss, global_step):
                pass
            cuml_loss += loss.item()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("loss/train", loss.item(), global_step)
            self.writer.add_scalar('extras/lr', self.optim_schedule.get_last_lr()[0], global_step)
            return cuml_loss / nb_iters

    def validate(self, global_step: int = None, verbose: bool = True):
        self.model.eval()

        preds = []
        loss_list = []
        for (x, y, z) in self.dl_valid:
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


def inspect_loss(loss, global_step: int):
    if torch.isnan(loss).sum().item():
        print("global step = {} ### nan encountered in loss. skipping".format(global_step))
        return True
    else:
        return False
