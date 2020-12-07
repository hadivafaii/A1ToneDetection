from .common import *
from .configuration import VAEConfig


class VAE(nn.Module):
    def __init__(self, config: VAEConfig, verbose: bool = False):
        super(VAE, self).__init__()

        self.config = config
        self.beta = 1.0

        self.embedding = Embedding(config, verbose=verbose)
        self.encoder = Encoder(config, verbose=verbose)
        self.decoder = Decoder(config, verbose=verbose)
        self.condition_inf = nn.ModuleList([ConditionINF(config, level=i) for i in range(config.nb_levels)])

        self.criterion_dff = NormalizedMSE(mode='var', reduction='sum')
        self.criterion_lick = nn.BCEWithLogitsLoss(reduction='sum')
        self.criterion_l = nn.CrossEntropyLoss(reduction='sum')
        self.criterion_f = nn.CrossEntropyLoss(reduction='sum')
        self.criterion_n = nn.CrossEntropyLoss(reduction='sum')

        self.apply(get_init_fn(config.init_range))
        if config.normalization == 'weight':
            self.apply(add_wn)
        elif config.normalization == 'spectral':
            self.apply(add_sn)

        if verbose:
            print_num_params(self)

    def forward(self, inputs_dict):
        output_z = ()
        inf_mu, inf_logsigma = (), ()
        gen_mu, gen_logsigma = (), ()
        output_dff, output_lick, output_label = (), (), ()

        # embed & encode
        embedded = self.embedding(inputs_dict)
        encoded_dff, encoded_lick = self.encoder(embedded['dff'], embedded['lick'])

        # compute top z
        z, mu, logsigma = self.forward_top_z(encoded_dff[-1], encoded_lick[-1], embedded['label'])
        output_z += (z,)
        inf_mu += (mu,)
        inf_logsigma += (logsigma,)

        # compute predictions from top level
        dff_y, lick_y, label_y = self.decoder.expand[0](z)
        dff_y = self.decoder.layers_dff[0](dff_y)
        lick_y = self.decoder.layers_lick[0](lick_y)
        output_dff += (dff_y,)
        output_lick += (lick_y,)
        output_label += (label_y,)

        for i in range(1, self.config.nb_levels):
            reverse_i = self.config.nb_levels - i - 1

            dff_y, lick_y, label_y = output_dff[i-1], output_lick[i-1], output_label[i-1]
            intermediate_inputs_dict = {'dff_y': dff_y, 'lick_y': lick_y, 'label_y': label_y}
            mu_z, logsigma_z = self.decoder.condition_gen[i-1](intermediate_inputs_dict)
            gen_mu += (mu_z,)
            gen_logsigma += (logsigma_z,)

            intermediate_inputs_dict['dff_x'] = encoded_dff[reverse_i]
            intermediate_inputs_dict['lick_x'] = encoded_lick[reverse_i]
            intermediate_inputs_dict['label_x'] = embedded['label']
            mu_xz, logsigma_xz = self.condition_inf[i](intermediate_inputs_dict)
            inf_mu += (mu_xz,)
            inf_logsigma += (logsigma_xz,)

            if self.config.residual_kl:
                z = reparametrize(mu_z + mu_xz, logsigma_z + logsigma_xz)
            else:
                z = reparametrize(mu_xz, logsigma_xz)
            output_z += (z,)

            dff_res, lick_res, label_res = self.decoder.expand[i](z)
            dff_y = self.decoder.layers_dff[i](dff_y + dff_res)
            lick_y = self.decoder.layers_lick[i](lick_y + lick_res)
            label_y = label_y + label_res
            output_dff += (dff_y,)
            output_lick += (lick_y,)
            output_label += (label_y,)

        # all outputs
        outputs_dict = {
            'dff': encoded_dff,
            'lick': encoded_lick,
            'label': embedded['label'],
            'z': output_z,
            'inf_mu': inf_mu,
            'inf_logsigma': inf_logsigma,
            'gen_mu': gen_mu,
            'gen_logsigma': gen_logsigma,
            'output_dff': output_dff,
            'output_lick': output_lick,
            'output_label': output_label,
        }

        # make predictions
        predictions_dict = self.make_predictions(inputs_dict, outputs_dict)

        return outputs_dict, predictions_dict

    def forward_top_z(self, dff_x, lick_x, label_x):
        dff_size, lick_size, label_size = tuple(map(lambda x: list(x.size()), [dff_x, lick_x, label_x]))
        dff_size[1] = 0
        lick_size[1] = 0
        label_size[1] = 0

        intermediate_inputs_dict = {
            'dff_x': dff_x,
            'dff_y': torch.empty(*dff_size).to(dff_x.device),
            'lick_x': lick_x,
            'lick_y': torch.empty(*lick_size).to(lick_x.device),
            'label_x': label_x,
            'label_y': torch.empty(*label_size).to(label_x.device),
        }

        mu0, logsigma0 = self.condition_inf[0](intermediate_inputs_dict)
        z0 = reparametrize(mu0, logsigma0)
        return z0, mu0, logsigma0

    def make_predictions(self, inputs_dict, outputs_dict):
        pred_dff_hidden = self.decoder.layers_dff[-1](outputs_dict['output_dff'][-1])
        pred_dffs = self.embedding.cell_embedding(pred_dff_hidden, inputs_dict['name'], encoding=False)
        pred_lick = self.decoder.layers_lick[-1](outputs_dict['output_lick'][-1])
        pred_l, pred_f, pred_n, indxs_l, indxs_f, indxs_n = self.predict_labels(outputs_dict['output_label'])

        predictions_dict = {
            'dff': pred_dffs,
            'lick': pred_lick,
            'label': pred_l,
            'freq': pred_f,
            'name': pred_n,
            'indxs_l': indxs_l,
            'indxs_f': indxs_f,
            'indxs_n': indxs_n,
        }
        return predictions_dict

    def predict_labels(self, output_label):
        pred_l, pred_f, pred_n = (), (), ()
        for i in range(self.config.nb_levels):
            pred_l += (output_label[i] @ self.embedding.label_embedding.weight.T,)
            pred_f += (output_label[i] @ self.embedding.stim_embedding.weight.T,)
            pred_n += (output_label[i] @ self.embedding.name_embedding.weight.T,)

        mean_pred_l = torch.cat([item.unsqueeze(0) for item in pred_l]).mean(0)
        mean_pred_f = torch.cat([item.unsqueeze(0) for item in pred_f]).mean(0)
        mean_pred_n = torch.cat([item.unsqueeze(0) for item in pred_n]).mean(0)
        indxs_l = torch.argmax(mean_pred_l, dim=1)
        indxs_f = torch.argmax(mean_pred_f, dim=1)
        indxs_n = torch.argmax(mean_pred_n, dim=1)

        return pred_l, pred_f, pred_n, indxs_l, indxs_f, indxs_n

    def compute_loss(self, inputs_dict, outputs_dict, predictions_dict, coeffs: Dict[str, float]):
        batch_size = inputs_dict['name'].shape[0]

        # update class weights to get balanced loss
        self._update_class_weights(inputs_dict)

        # reconstruction
        # continuous (normalized by nb_timepoints)
        loss_dff = sum([
            self.criterion_dff(pred, target) / self.config.nb_timepoints / batch_size
            for pred, target in zip(predictions_dict['dff'], inputs_dict['dff'])
        ])
        loss_lick = self.criterion_lick(
            predictions_dict['lick'], inputs_dict['lick']) / self.config.nb_timepoints / batch_size
        # categorical # TODO: do I normalize these as well?
        loss_l = tuple(self.criterion_l(pred, inputs_dict['label']) / batch_size      # / len(self.config.l2i)
                       for pred in predictions_dict['label'])
        loss_f = tuple(self.criterion_l(pred, inputs_dict['freq']) / batch_size     # / len(self.config.f2i)
                       for pred in predictions_dict['freq'])
        loss_n = tuple(self.criterion_l(pred, inputs_dict['name']) / batch_size     # / len(self.config.n2i)
                       for pred in predictions_dict['name'])

        # kl
        loss_kl = self.compute_kl_loss(
            inf_mu=outputs_dict['inf_mu'],
            inf_logsigma=outputs_dict['inf_logsigma'],
            gen_mu=outputs_dict['gen_mu'],
            gen_logsigma=outputs_dict['gen_logsigma'],
        )

        # all losses
        loss_dict = {
            'dff': loss_dff,
            'lick': loss_lick,
            'label': loss_l,
            'freq': loss_f,
            'name': loss_n,
            'kl': loss_kl,
        }
        recon_loss = (coeffs['dff'] * loss_dff +
                      coeffs['lick'] * loss_lick +
                      coeffs['label'] * sum(x for x in loss_l) +
                      coeffs['freq'] * sum(x for x in loss_f) +
                      coeffs['name'] * sum(x for x in loss_n)
                      )
        loss = recon_loss + self.beta * sum(x for x in loss_kl)
        return loss, loss_dict

    def compute_kl_loss(self, inf_mu, inf_logsigma, gen_mu, gen_logsigma):
        batch_size = inf_mu[0].shape[0]
        kl = ()

        # top_level
        logsigma = torch.zeros_like(inf_mu[0]).to(inf_mu[0].device)
        loss = gaussian_residual_kl(inf_mu[0], inf_logsigma[0], logsigma)
        kl += (loss / batch_size,)

        # mid levels
        for i in range(1, self.config.nb_levels):
            if self.config.residual_kl:
                loss = gaussian_residual_kl(inf_mu[i], inf_logsigma[i], gen_logsigma[i-1])
            else:
                loss = gaussian_analytical_kl(inf_mu[i], gen_mu[i-1], inf_logsigma[i], gen_logsigma[i-1])
            kl += (loss / batch_size,)

        return kl

    def _update_class_weights(self, inputs_dict):
        pass

    def update_beta(self, new_beta: float):
        self.beta = max(new_beta, 0.0)


class Decoder(nn.Module):
    def __init__(self, config: VAEConfig, verbose: bool = False):
        super(Decoder, self).__init__()

        self.layers_dff = nn.ModuleList(self._make_net(config, mode='dff'))
        self.layers_lick = nn.ModuleList(self._make_net(config, mode='lick'))
        self.condition_gen = nn.ModuleList([ConditionGEN(config, level=i) for i in range(1, config.nb_levels)])
        self.expand = nn.ModuleList([Expand(config, level=i) for i in range(config.nb_levels)])

        if verbose:
            print_num_params(self)

    # TODO: fix this later
    def forward(self):
        pass

    @staticmethod
    def _make_net(config, mode):
        if mode == 'dff':
            init_dim = config.h_dim
            final_dim = config.cell_embedding_dim
        elif mode == 'lick':
            init_dim = config.lick_embedding_dim
            final_dim = 1
        else:
            raise NotImplementedError

        num_channels = [init_dim * 2**i for i in range(config.nb_levels + 1)]
        num_channels = num_channels[::-1]
        assert len(num_channels) == config.nb_levels + 1

        layers = []
        for i in range(config.nb_levels):
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [nn.Sequential(
                TemporalBlockTransposed(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=config.kernel_size,
                    activation_fn=config.activation_fn,
                    dropout=config.dropout,
                    bias=config.use_bias,
                    linear_output=False),
                nn.Upsample(
                    size=config.hierarchy_size[i+1],
                    mode=config.upsample_mode,
                    align_corners=False),
            )]
        layers += [nn.Sequential(
            TemporalBlockTransposed(
                n_inputs=num_channels[-1],
                n_outputs=final_dim,
                kernel_size=config.kernel_size,
                activation_fn=config.activation_fn,
                dropout=config.dropout,
                bias=config.use_bias,
                linear_output=True),
            Permute(dims=(0, 2, 1)),
        )]
        return layers


class Expand(nn.Module):
    def __init__(self, config: VAEConfig, level: int):
        super(Expand, self).__init__()

        assert 0 <= level <= config.nb_levels
        self.level = level

        outplanes_dff = config.h_dim * 2 ** (config.nb_levels - self.level)
        outplanes_lick = config.lick_embedding_dim * 2 ** (config.nb_levels - self.level)
        intermediate_size = config.hierarchy_size[self.level]   # TODO: here can also use upsample, but probably not...
        self.deconv_dff = nn.ConvTranspose1d(config.z_dim, outplanes_dff, intermediate_size, bias=config.use_bias)
        self.deconv_lick = nn.ConvTranspose1d(config.z_dim, outplanes_lick, intermediate_size, bias=config.use_bias)
        self.expand_label = nn.Linear(config.z_dim, config.label_embedding_dim, bias=True)   # TODO: config.use_bias?

    def forward(self, z):
        label = self.expand_label(z)
        z = z.unsqueeze(-1)
        dff = self.deconv_dff(z)
        lick = self.deconv_lick(z)
        return dff, lick, label


class Condition(nn.Module):
    def __init__(self, config: VAEConfig, mode: str, level: int):
        super(Condition, self).__init__()

        assert mode in ['inference', 'generation']
        assert 0 <= level <= config.nb_levels
        self.mode = mode
        self.level = level

        self.inplanes = config.planes[self.level]
        self.inplanes *= 2 if (self.mode == 'inference' and self.level > 0) else 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(self.inplanes, config.z_dim * 2, kernel_size=1, bias=True)

    def forward(self, x):
        return self.condition(x)

    @staticmethod
    def condition(x):
        mu, logsigma = x.squeeze().chunk(2, dim=-1)
        return mu, logsigma


class ConditionINF(Condition):
    def __init__(self, config: VAEConfig, level: int):
        super(ConditionINF, self).__init__(config, mode='inference', level=level)

    def forward(self, inputs_dict):
        x = [
            inputs_dict['dff_x'], inputs_dict['dff_y'],
            inputs_dict['lick_x'], inputs_dict['lick_y'],
        ]
        x = torch.cat(x, dim=1)
        x = self.pool(x)
        x = [x, inputs_dict['label_x'].unsqueeze(2), inputs_dict['label_y'].unsqueeze(2)]
        x = torch.cat(x, dim=1)
        x = self.conv(x)

        mu, logsigma = self.condition(x)
        return mu, logsigma   # TODO: retun x as well?


class ConditionGEN(Condition):
    def __init__(self, config: VAEConfig, level: int):
        super(ConditionGEN, self).__init__(config, mode='generation', level=level)

    def forward(self, inputs_dict):
        x = [inputs_dict['dff_y'], inputs_dict['lick_y']]
        x = torch.cat(x, dim=1)
        x = self.pool(x)
        x = [x, inputs_dict['label_y'].unsqueeze(2)]
        x = torch.cat(x, dim=1)
        x = self.conv(x)

        mu, logsigma = self.condition(x)
        return mu, logsigma   # TODO: retun x as well?


class Encoder(nn.Module):
    def __init__(self, config: VAEConfig, verbose: bool = False):
        super(Encoder, self).__init__()

        self.tcn_dff = TCN(config, init_dim=config.h_dim, verbose=verbose)
        self.tcn_lick = TCN(config, init_dim=config.lick_embedding_dim, verbose=verbose)

        if verbose:
            print_num_params(self)

    def forward(self, dff, lick):
        dff = self.tcn_dff(dff)
        lick = self.tcn_lick(lick)
        return dff, lick


class Embedding(nn.Module):
    def __init__(self, config, verbose=False):
        super(Embedding, self).__init__()

        self.label_embedding = nn.Embedding(
            num_embeddings=len(config.l2i),
            embedding_dim=config.label_embedding_dim,)
        self.stim_embedding = nn.Embedding(
            num_embeddings=len(config.f2i),
            embedding_dim=config.label_embedding_dim,)
        self.name_embedding = nn.Embedding(
            num_embeddings=len(config.n2i),
            embedding_dim=config.label_embedding_dim,)

        self.lick_embedding = nn.Sequential(
            nn.Linear(1, config.lick_embedding_dim, bias=False),
            Permute(dims=(0, 2, 1)),
        )
        self.cell_embedding = CellEmbedding(config, verbose)

        # TODO: maybe add gain and bias here?
        # TODO: Gain is a function of labels, for example it's weight is (16 x nc)

        if verbose:
            print_num_params(self)

    def forward(self, inputs_dict):
        dff_embedded = self.cell_embedding(inputs_dict['dff'], inputs_dict['name'])
        lick_embedded = self.lick_embedding(inputs_dict['lick'])
        l_emb = self.label_embedding(inputs_dict['label'])
        f_emb = self.stim_embedding(inputs_dict['freq'])
        n_emb = self.name_embedding(inputs_dict['name'])
        label_embedded = l_emb + f_emb + n_emb

        return dict(dff=dff_embedded, lick=lick_embedded, label=label_embedded)


class CellEmbedding(nn.Module):
    def __init__(self, config, verbose=False):
        super(CellEmbedding, self).__init__()

        self.layers = nn.ModuleDict(
            {str(config.n2i[name]): nn.Linear(nc, config.cell_embedding_dim, bias=False)
             for name, nc in config.nb_cells.items()}
        )
        self.biases = nn.ParameterDict(
            {str(config.n2i[name]): nn.Parameter(torch.zeros(nc, dtype=torch.float))
             for name, nc in config.nb_cells.items()}
        )
        # self.gains = nn.ParameterDict(
        #     {str(config.n2i[name]): nn.Parameter(torch.ones(nc, dtype=torch.float))
        #      for name, nc in config.nb_cells.items()}
        # )
        self.mapping_out = nn.Sequential(
            nn.Linear(config.cell_embedding_dim, config.h_dim, bias=False),
            Permute(dims=(0, 2, 1)),
        )

        if verbose:
            print_num_params(self)

    def forward(self, inputs, names, encoding: bool = True):
        output = (
            F.linear(x, self.layers[str(n.item())].weight, None).unsqueeze(0) if encoding else
            F.linear(x, self.layers[str(n.item())].weight.T, self.biases[str(n.item())])    # Gain?
            for x, n in zip(inputs, names)
        )
        return self.mapping_out(torch.cat(list(output))) if encoding else list(output)


class TemporalConvNet(nn.Module):
    def __init__(self, config: VAEConfig, init_dim: int, verbose: bool = False):
        super(TemporalConvNet, self).__init__()

        num_channels = [init_dim * 2**i for i in range(config.nb_levels + 1)]
        assert len(num_channels) == config.nb_levels + 1
        layers = []
        for i in range(config.nb_levels):
            dilation_size = 2 ** i if config.use_dilation else 1
            in_channels = num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=config.kernel_size,
                dilation=dilation_size,
                padding=(config.kernel_size-1) * dilation_size,
                activation_fn=config.activation_fn,
                dropout=config.dropout,
                bias=config.use_bias,
            )]
        self.layers = nn.ModuleList(layers)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        output = ()
        for layer in self.layers:
            x = layer(x)
            output += (x,)
        return output


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, activation_fn, dropout=0.2, bias=False):
        super(TemporalBlock, self).__init__()
        conv1 = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        chomp1 = Chomp1d(padding)
        dropout1 = nn.Dropout(dropout, inplace=True)

        conv2 = nn.Conv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        chomp2 = Chomp1d(padding//2)
        dropout2 = nn.Dropout(dropout, inplace=True)

        self.net = nn.Sequential(
            conv1, chomp1, get_activation_fn(activation_fn), dropout1,
            conv2, chomp2, get_activation_fn(activation_fn), dropout2,
        )
        self.downsample = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=1,
            stride=2,
            bias=bias,) if n_inputs != n_outputs else None
        self.activation = get_activation_fn(activation_fn)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalBlockTransposed(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, activation_fn, dropout=0.0, bias=False, linear_output=False):
        super(TemporalBlockTransposed, self).__init__()
        deconv1 = nn.ConvTranspose1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
        )
        chomp1 = Chomp1d(kernel_size-1)
        dropout1 = nn.Dropout(dropout, inplace=True)

        deconv2 = nn.ConvTranspose1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
        )
        chomp2 = Chomp1d(kernel_size-1)
        dropout2 = nn.Dropout(dropout, inplace=True)

        if linear_output:
            self.net = nn.Sequential(
                deconv1, chomp1, get_activation_fn(activation_fn), dropout1,
                deconv2, chomp2,
            )
        else:
            self.net = nn.Sequential(
                deconv1, chomp1, get_activation_fn(activation_fn), dropout1,
                deconv2, chomp2, get_activation_fn(activation_fn), dropout2,
            )
        self.downsample = nn.ConvTranspose1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=1,
            stride=1,
            bias=bias,) if n_inputs != n_outputs else None
        self.activation = get_activation_fn(activation_fn) if not linear_output else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res) if self.activation is not None else out + res


CI = ConditionINF
CG = ConditionGEN
EX = Expand
TCN = TemporalConvNet
