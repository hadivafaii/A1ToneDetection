import os
import h5py
import numpy as np
from typing import List, Dict
from os.path import join as pjoin


class BaseConfig(object):
    def __init__(self,
                 include_trials: List[str] = None,
                 include_freqs: List[int] = None,
                 l2i: Dict[str, int] = None,
                 i2l: Dict[int, str] = None,
                 nb_cells: Dict[str, int] = None,
                 nb_std: int = 1,
                 base_dir: str = 'Documents/A1',
                 h_file: str = None,):
        super(BaseConfig, self).__init__()
        # trial types and frequencies to include in analysis
        include_trials_default = ['hit', 'miss', 'correctreject', 'falsealarm', 'passive']
        include_freqs_default = [7000, 9899, 14000, 19799, 7071, 8000, 10000, 14142, 20000, 22627]
        self.include_trials = include_trials_default if include_trials is None else include_trials
        self.include_freqs = include_freqs_default if include_freqs is None else include_freqs
        self.l2i = {} if l2i is None else l2i
        self.i2l = {} if i2l is None else i2l
        self.set_l2i2l()

        # dir configs
        self.nb_std = nb_std
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        _processed_dir = pjoin(self.base_dir, 'python_processed')
        _file_name = "organized_nb_std={:d}.h5".format(nb_std)
        self.h_file = pjoin(_processed_dir, _file_name) if h_file is None else h_file

        # nb_cells dict
        self.nb_cells = {}
        self.set_nb_cells(nb_cells)

    def set_nb_cells(self, nb_cells: Dict[str, int] = None):
        if nb_cells is not None:
            self.nb_cells = nb_cells
        else:
            nb_cells = {}
            f = h5py.File(self.h_file, 'r')
            for name in f:
                behavior = f[name]['behavior']
                passive = f[name]['passive']
                good_cells_b = np.array(behavior["good_cells"], dtype=int)
                good_cells_p = np.array(passive["good_cells"], dtype=int)
                good_cells = set(good_cells_b).intersection(set(good_cells_p))
                good_cells = list(good_cells)
                nb_cells[name] = len(good_cells)
            self.nb_cells = nb_cells
            f.close()

    def set_l2i2l(self):
        if not len(self.l2i):
            self.l2i = {lbl: i for i, lbl in enumerate(self.include_trials)}
        if not len(self.i2l):
            self.i2l = {i: lbl for lbl, i in self.l2i.items()}


class FeedForwardConfig(BaseConfig):
    def __init__(self,
                 h_dim: int = 64,
                 z_dim: int = 16,
                 c_dim: int = 8,
                 time_slice: range = range(30, 45),
                 loss_lambda: float = 1.0,
                 embedding_dropout: float = 0.2,
                 classifier_dropout: float = 0.5,
                 **kwargs,):
        super(FeedForwardConfig, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.time_slice = time_slice
        self.loss_lambda = loss_lambda
        self.embedding_dropout = embedding_dropout
        self.classifier_dropout = classifier_dropout


class TrainConfig:
    def __init__(self,
                 lr: float = 1e-2,
                 weight_decay: float = 1e-1,
                 scheduler_period: int = 100,
                 eta_min: float = 1e-8,
                 batch_size: int = 64,

                 balanced_sampling: bool = True,
                 replacement: bool = False,

                 log_freq: int = 100,
                 chkpt_freq: int = 1,
                 eval_freq: int = 5,
                 random_state: int = 42,
                 xv_folds: int = 5,
                 use_cuda: bool = True,
                 runs_dir: str = 'Documents/A1/runs',):

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_period = scheduler_period
        self.eta_min = eta_min
        self.batch_size = batch_size

        self.balanced_sampling = balanced_sampling
        self.replacement = replacement

        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.eval_freq = eval_freq
        self.random_state = random_state
        self.xv_folds = xv_folds
        self.use_cuda = use_cuda
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
