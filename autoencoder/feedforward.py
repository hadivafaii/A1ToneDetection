from torch.nn.utils import weight_norm

from .common import *
from .model_utils import print_num_params
from .configuration import FeedForwardConfig


class TiedAutoEncoder(nn.Module):
    def __init__(self,
                 config: FeedForwardConfig,
                 verbose: bool = False,):
        super(TiedAutoEncoder, self).__init__()
        self.config = config
        self.embedding = CellEmbedding(config, verbose)
        self.encoder = nn.Sequential(
            weight_norm(nn.Linear(config.h_dim, config.z_dim, bias=True)),
            nn.BatchNorm1d(config.z_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            weight_norm(nn.Linear(config.z_dim, config.h_dim, bias=True)),
            nn.BatchNorm1d(config.h_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(config.embedding_dropout)
        self.criterion = NormalizedMSE(mode='var', dim=0)

        if verbose:
            print_num_params(self)

    def forward(self, name, x):
        x = self.embedding(name, x, encoding=True)
        x = self.dropout(x)
        z = self.encoder(x)
        y = self.decoder(z)
        y = self.embedding(name, y, encoding=False)
        return y, z


class Classifier(nn.Module):
    def __init__(self,
                 config: FeedForwardConfig,
                 verbose: bool = False,):
        super(Classifier, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.h_dim, config.c_dim, bias=True)
        self.fc2 = nn.Linear(config.c_dim, len(config.l2i), bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.GRU(
            input_size=config.c_dim,
            hidden_size=config.c_dim,
            batch_first=True,
            bias=True,)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        if verbose:
            print_num_params(self)

    def forward(self, x):
        if len(x.size()) == 3:
            slice_ = slice(self.config.start_time, self.config.end_time, 1)
            x = x[:, slice_, :].contiguous()
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        _, h = self.rnn(x)
        y = self.fc2(h.squeeze())
        return y


class AutoEncoder(nn.Module):
    def __init__(self,
                 config,
                 nb_cells,
                 verbose=False,):
        super(AutoEncoder, self).__init__()

        self.encoder = FFEncoder(config, nb_cells, verbose)
        self.decoder = FFDecoder(config, nb_cells, verbose)
        self.criterion = nn.MSELoss(reduction="sum")

        if verbose:
            print_num_params(self)

    def forward(self, name, x):
        z = self.encoder(name, x)
        y = self.decoder(name, z)
        return z, y


class FFEncoder(nn.Module):
    def __init__(self,
                 config,
                 nb_cells,
                 verbose=False,):
        super(FFEncoder, self).__init__()

        self.fc1 = nn.ModuleDict(
            {
                name: nn.Linear(nc, config.h_dim, bias=True)
                for name, nc in nb_cells.items()
            }
        )
        self.fc2 = nn.Linear(config.h_dim, config.z_dim, bias=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        if verbose:
            print_num_params(self)

    def forward(self, name, x):
        x = self.fc1[name](x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class FFDecoder(nn.Module):
    def __init__(self,
                 config,
                 nb_cells,
                 verbose=False,):
        super(FFDecoder, self).__init__()

        self.fc1 = nn.Linear(config.z_dim, config.h_dim, bias=True)
        self.fc2 = nn.ModuleDict(
            {
                name: nn.Linear(config.h_dim, nc, bias=True)
                for name, nc in nb_cells.items()
            }
        )
        self.relu = nn.ReLU(inplace=True)

        if verbose:
            print_num_params(self)

    def forward(self, name, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2[name](x)
        return x
