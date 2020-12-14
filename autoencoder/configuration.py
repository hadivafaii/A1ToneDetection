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
                 f2i: Dict[int, int] = None,
                 i2f: Dict[int, int] = None,
                 n2i: Dict[str, int] = None,
                 i2n: Dict[int, str] = None,
                 nb_cells: Dict[str, int] = None,
                 nb_timepoints: int = 135,
                 init_range: float = 0.01,
                 nb_std: int = 1,
                 base_dir: str = 'Documents/A1',
                 h_file: str = None,):
        super(BaseConfig, self).__init__()
        # trial types and frequencies to include in analysis
        include_trials_default = ['hit', 'miss', 'correctreject', 'falsealarm', 'passive']
        # include_freqs_default = [
        #     4000, 5000, 5657, 7071, 8000, 10000, 11314, 14142,
        #     16000, 20000, 22627, 28284, 32000, 40000, 45255, 56569,
        # ]
        include_freqs_default = [5000, 7071, 10000, 14142, 20000, 28284]  # only passive
        self.include_trials = include_trials_default if include_trials is None else include_trials
        self.include_freqs = include_freqs_default if include_freqs is None else include_freqs

        # other
        self.nb_timepoints = nb_timepoints
        self.init_range = init_range

        # dir configs
        self.nb_std = nb_std
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        _processed_dir = pjoin(self.base_dir, 'python_processed')
        _file_name = "organized_nb_std={:d}.h5".format(nb_std)
        self.h_file = pjoin(_processed_dir, _file_name) if h_file is None else h_file

        # nb_cells dict
        self.nb_cells = nb_cells
        self._set_nb_cells()

        # lookup dicts
        self.l2i = {} if l2i is None else l2i
        self.i2l = {} if i2l is None else i2l
        self.f2i = {} if f2i is None else f2i
        self.i2f = {} if i2f is None else i2f
        self.n2i = {} if n2i is None else n2i
        self.i2n = {} if i2n is None else i2n
        self._set_lookup_dicts()

    def _set_nb_cells(self):
        if self.nb_cells is not None:
            pass
        else:
            nb_cells = {}
            f = h5py.File(self.h_file, 'r')
            for name in f:
                behavior = f[name]['behavior']
                passive = f[name]['passive']
                good_cells_b = np.array(behavior["good_cells"], dtype=int)
                good_cells_p = np.array(passive["good_cells"], dtype=int)
                good_cells = set(good_cells_b).intersection(set(good_cells_p))
                good_cells = sorted(list(good_cells))
                nb_cells[name] = len(good_cells)
            self.nb_cells = nb_cells
            f.close()

    def _set_lookup_dicts(self):
        # labels
        if not len(self.l2i):
            self.l2i = {lbl: i for i, lbl in enumerate(self.include_trials)}
            self.i2l = {i: lbl for lbl, i in self.l2i.items()}

        # stim freq
        if not len(self.f2i):
            f2i = {freq: i for i, freq in enumerate(self.include_freqs)}
            behavior_freqs = {
                7000: f2i[7071],
                9899: f2i[10000],
                14000: f2i[14142],
                19799: f2i[20000],
            }
            f2i.update(behavior_freqs)
            self.f2i = dict(sorted(f2i.items()))
            self.i2f = {self.f2i[freq]: freq for freq in self.include_freqs}

        # expt names
        if not len(self.n2i):
            self.n2i = {name: i for i, name in enumerate(self.nb_cells.keys())}
            self.i2n = {i: name for name, i in self.n2i.items()}


class VAEConfig(BaseConfig):
    def __init__(self,
                 lick_embedding_dim: int = 4,
                 label_embedding_dim: int = 16,
                 cell_embedding_dim: int = 64,
                 # h_dim: int = 32,
                 z_dim: int = 8,
                 nb_levels: int = 4,
                 kernel_size: int = 3,

                 activation_fn: str = 'relu',
                 upsample_mode: str = 'linear',
                 normalization: str = 'spectral',
                 residual_kl: bool = True,
                 use_dilation: bool = False,
                 use_bias: bool = False,
                 dropout: float = 0.0,

                 planes: Dict[int, int] = None,
                 hierarchy_size: Dict[int, int] = None,

                 **kwargs,
                 ):
        super(VAEConfig, self).__init__(**kwargs)

        self.lick_embedding_dim = lick_embedding_dim
        self.label_embedding_dim = label_embedding_dim
        self.cell_embedding_dim = cell_embedding_dim
        # self.h_dim = h_dim
        self.z_dim = z_dim
        self.nb_levels = nb_levels
        self.kernel_size = kernel_size

        _allowed_activation_fn = ['relu', 'swish', 'learned_swish', 'gelu']
        assert activation_fn in _allowed_activation_fn,\
            "allowed scheduler types: {}".format(_allowed_activation_fn)
        self.activation_fn = activation_fn
        self.upsample_mode = upsample_mode
        self.normalization = normalization
        self.residual_kl = residual_kl
        self.use_dilation = use_dilation
        self.use_bias = use_bias
        self.dropout = dropout

        self.planes = {} if planes is None else planes
        self.hierarchy_size = {} if hierarchy_size is None else hierarchy_size
        self._compute_hierarchy_dims()

    def _compute_hierarchy_dims(self):
        for level in range(self.nb_levels + 1):
            i = self.nb_levels - level
            self.planes[level] = (self.cell_embedding_dim + self.lick_embedding_dim) * 2 ** i + self.label_embedding_dim
            self.hierarchy_size[level] = int(np.ceil(self.nb_timepoints / 2 ** i))


class FeedForwardConfig(BaseConfig):
    def __init__(self,
                 h_dim: int = 64,
                 z_dim: int = 16,
                 c_dim: int = 8,

                 start_time: int = 30,
                 end_time: int = 45,

                 loss_lambda: float = 1.0,
                 embedding_dropout: float = 0.2,
                 classifier_dropout: float = 0.5,
                 **kwargs,
                 ):
        super(FeedForwardConfig, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.start_time = start_time
        self.end_time = end_time

        self.loss_lambda = loss_lambda
        self.embedding_dropout = embedding_dropout
        self.classifier_dropout = classifier_dropout


class TrainConfig:
    def __init__(self,
                 lr: float = 1e-3,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 batch_size: int = 64,
                 weight_decay: float = 1e-2,

                 scheduler_period: int = 20,
                 eta_min: float = 1e-8,
                 scheduler_type: str = 'cosine',
                 scheduler_gamma: float = 0.9,

                 loss_coeffs: Dict[str, float] = None,
                 grad_clip: float = 200.0,
                 skip_threshold: float = 300.0,
                 beta_warmup_steps: int = int(2e4),
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.scheduler_period = scheduler_period
        self.eta_min = eta_min

        _allowed_schedulers = ['cosine', 'exponential', 'step', None]
        assert scheduler_type in _allowed_schedulers,\
            "allowed scheduler types: {}".format(_allowed_schedulers)
        self.scheduler_type = scheduler_type
        self.scheduler_gamma = scheduler_gamma

        _loss_coeff_defaults = {
            'dff': 1.0,
            'lick': 5.0,
            'label': 1.0,
            'freq': 1.0,
            'name': 1.0,
        }
        self.loss_coeffs = _loss_coeff_defaults if loss_coeffs is None else loss_coeffs
        self.grad_clip = grad_clip
        self.skip_threshold = skip_threshold
        self.beta_warmup_steps = beta_warmup_steps
        self.balanced_sampling = balanced_sampling
        self.replacement = replacement

        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.eval_freq = eval_freq
        self.random_state = random_state
        self.xv_folds = xv_folds
        self.use_cuda = use_cuda
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
