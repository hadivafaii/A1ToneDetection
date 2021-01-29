import sys
import h5py
import rcca
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

sys.path.append('..')
from utils.generic_utils import *
from tqdm.notebook import tqdm
from scipy.stats import zscore
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit_cca_loop(
        h_load_file: str,
        rescale: bool = False,
        save_file: str = None,
        **kwargs, ):
    save_file = 'results_cca_{}.df'.format(now()) if save_file is None else save_file
    default_args = {
        'min_nb_trials': 100,
        'target': True,
        'global_normalize': True,
        'augment_data': False,
        'xv_folds': 5,
        'time_range': range(45, 46),
        'num_ccs': np.arange(5, 91, 5),
        'cca_regs': np.logspace(-3, -1.5, num=20),
        'cutoffs': np.logspace(-18, -12, num=3),
    }
    for k in default_args:
        if k in kwargs:
            default_args[k] = kwargs[k]

    results = pd.DataFrame()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for fold in tqdm(range(default_args['xv_folds']), leave=False):
        data_trn, data_tst = prepare_cca_data(
            h_load_file=h_load_file,
            min_nb_trials=default_args['min_nb_trials'],
            target=default_args['target'],
            global_normalize=default_args['global_normalize'],
            augment_data=default_args['augment_data'],
            xv_folds=default_args['xv_folds'],
            which_fold=fold,
            time_range=default_args['time_range'],
            verbose=False,
        )
        train_list, y_trn = data_trn['processed'], data_trn['labels']
        test_list, y_tst = data_tst['processed'], data_tst['labels']

        for n_components in tqdm(default_args['num_ccs'], leave=False):
            train_list_centered = [
                (item - item.mean()) / np.sqrt(n_components) if rescale else item - item.mean()
                for item in train_list
            ]
            for reg in tqdm(default_args['cca_regs'], leave=False):
                for cutoff in tqdm(default_args['cutoffs'], leave=False):
                    cca = rcca.CCA(
                        kernelcca=True,
                        ktype='linear',
                        numCC=n_components,
                        reg=reg,
                        cutoff=cutoff,
                        verbose=False,
                    )
                    cca.train(train_list_centered)
                    testcorrs = cca.validate(test_list)

                    corrs = []
                    for item in testcorrs:
                        corrs.append(np.mean(np.abs(item)))
                    pred_r = np.mean(corrs)

                    data_dict = {
                        'fold': [fold],
                        'n_components': [n_components],
                        'cca_reg': [reg],
                        'cutoff': [cutoff],
                        'metric': ['pred_r'],
                        'value': [pred_r],
                    }
                    results = pd.concat([results, pd.DataFrame.from_dict(data_dict)])
            save_obj(obj=results, file_name=save_file, save_dir='./results', mode='df', verbose=False)

    results = reset_df(results, downcast='none')
    save_obj(obj=results, file_name=save_file, save_dir='./results', mode='df', verbose=True)
    # TODO: reimplement extract best hyperparams so that it works for this too
    return results, default_args


def get_best_cca_clf(
        h_load_file: str,
        best: dict,
        min_nb_trials: int = -1,
        time_range: range = range(45, 46),
        target: bool = True,
        global_normalize: bool = True,
        augment_data: bool = False,
        xv_folds: int = 5,
        which_fold: int = 0,
        random_sate: int = 42, ):

    data_trn, data_tst = prepare_cca_data(
        h_load_file=h_load_file,
        min_nb_trials=min_nb_trials,
        target=target,
        global_normalize=global_normalize,
        augment_data=augment_data,
        xv_folds=xv_folds,
        which_fold=which_fold,
        time_range=time_range,
        verbose=False,
    )
    train_list, y_trn = data_trn['processed'], data_trn['labels']
    test_list, y_tst = data_tst['processed'], data_tst['labels']

    cca = rcca.CCA(
        kernelcca=True,
        ktype='linear',
        reg=best['cca_reg'],
        numCC=best['n_components'],
        verbose=False,
    )
    cca.train([item / np.sqrt(best['n_components']) for item in train_list])
    testcorrs = cca.validate(test_list)

    corrs = []
    for item in testcorrs:
        corrs.append(np.mean(np.abs(item)))
    pred_r = np.mean(corrs)

    x_trn = [x @ w for x, w in zip(train_list, cca.ws)]
    x_tst = [x @ w for x, w in zip(test_list, cca.ws)]
    x_trn, x_tst = tuple(map(np.concatenate, [x_trn, x_tst]))

    clf = LogisticRegression(
        random_state=random_sate,
        C=best['clf_reg'],
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        max_iter=int(1e4),
        tol=1e-6,
    ).fit(x_trn, y_trn)
    y_pred = clf.predict(x_tst)

    balacc = balanced_accuracy_score(y_tst, y_pred)
    mcc = matthews_corrcoef(y_tst, y_pred)

    msg = "[PROGRESS] fitting done. results:\n"
    msg += "corr: {:.3f},   balanced accuracy: {:.3f},   mcc: {:.3f}"
    msg = msg.format(pred_r, balacc, mcc)
    print(msg)

    comps_trn, comps_df_trn = extract_components(data_trn, cca)
    comps_tst, comps_df_tst = extract_components(data_tst, cca)

    output = {
        'data_trn': data_trn,
        'data_tst': data_tst,
        'cca': cca,
        'clf': clf,
        'comps_trn': comps_trn,
        'comps_tst': comps_tst,
        'comps_df_trn': comps_df_trn,
        'comps_df_tst': comps_df_tst,
    }
    return output


def extract_components(data: dict, cca):
    df = pd.DataFrame()
    components = []
    for idx, name in enumerate(data['raw'].keys()):
        comps = data['processed'][idx] @ cca.ws[idx]
        components.append(comps)

        data_dict = {'name': [name] * len(comps)}
        for cc in range(comps.shape[1]):
            data_dict.update({'component_{:d}'.format(cc): comps[:, cc]})
        df = pd.concat([df, pd.DataFrame.from_dict(data_dict)])
    return np.concatenate(components), reset_df(df)


def fit_cca_clf_loop(
        h_load_file: str,
        n_seeds: int = 3,
        rescale: bool = True,
        save_file: str = None,
        **kwargs, ):
    save_file = 'results_cca_clf_{}.df'.format(now()) if save_file is None else save_file
    default_args = {
        'seeds': [int(2 ** i) for i in range(max(1, n_seeds))],
        'min_nb_trials': 100,
        'target': True,
        'global_normalize': True,
        'augment_data': False,
        'xv_folds': 5,
        'timepoint': 45,
        'num_ccs': np.arange(5, 91, 5),
        'cca_regs': np.logspace(-3, -1.5, num=20),
        'clf_regs': np.logspace(-3, -1.4, num=20),
        'clf_max_iter': int(1e3),
        'clf_tol': 1e-4,
    }
    for k in default_args:
        if k in kwargs:
            default_args[k] = kwargs[k]

    results = pd.DataFrame()
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    for random_state in tqdm(default_args['seeds']):
        np.random.seed(random_state)

        for fold in tqdm(range(default_args['xv_folds']), leave=False):
            data_trn, data_tst = prepare_cca_data(
                h_load_file=h_load_file,
                min_nb_trials=default_args['min_nb_trials'],
                timepoint=default_args['timepoint'],
                target=default_args['target'],
                normalize_mode='zscore',
                augment=default_args['augment_data'],
                xv_folds=default_args['xv_folds'],
                which_fold=fold,
                verbose=False,
            )
            train_list, y_trn = data_trn['processed'], data_trn['labels']
            test_list, y_tst = data_tst['processed'], data_tst['labels']

            for n_components in tqdm(default_args['num_ccs'], leave=False):
                train_list_centered = [
                    (item - item.mean()) / np.sqrt(n_components) if rescale else item - item.mean()
                    for item in train_list
                ]
                for reg in tqdm(default_args['cca_regs'], leave=False):
                    cca = rcca.CCA(
                        kernelcca=True,
                        ktype='linear',
                        numCC=n_components,
                        reg=reg,
                        cutoff=1e-15,
                        verbose=False,
                    ).train(train_list_centered)
                    testcorrs = cca.validate(test_list)

                    corrs = []
                    for item in testcorrs:
                        corrs.append(np.mean(np.abs(item)))
                    pred_r = np.mean(corrs)

                    x_trn = [x @ w for x, w in zip(train_list, cca.ws)]
                    x_tst = [x @ w for x, w in zip(test_list, cca.ws)]
                    x_trn, x_tst = tuple(map(np.concatenate, [x_trn, x_tst]))

                    for C in default_args['clf_regs']:
                        clf = LogisticRegression(
                            C=C,
                            penalty='l1',
                            solver='liblinear',
                            class_weight='balanced',
                            max_iter=default_args['clf_max_iter'],
                            tol=default_args['clf_tol'],
                            random_state=random_state,
                        ).fit(x_trn, y_trn)
                        y_pred = clf.predict(x_tst)

                        balacc = balanced_accuracy_score(y_tst, y_pred)
                        mcc = matthews_corrcoef(y_tst, y_pred)

                        data_dict = {
                            'seed': [random_state] * 3,
                            'fold': [fold] * 3,
                            'n_components': [n_components] * 3,
                            'cca_reg': [reg] * 3,
                            'clf_reg': [C] * 3,
                            'metric': ['mcc', 'bal_acc', 'pred_r'],
                            'value': [mcc, balacc, pred_r],
                        }
                        results = pd.concat([results, pd.DataFrame.from_dict(data_dict)])
                save_obj(obj=results, file_name=save_file, save_dir='./results', mode='df', verbose=False)

    results = reset_df(results)
    save_obj(obj=results, file_name=save_file, save_dir='./results', mode='df', verbose=True)
    best = extract_best_hyperparams(results, metric='mcc', verbose=True)

    return results, best, default_args


def extract_best_hyperparams2(
        results: pd.DataFrame,
        metrics: List[str] = 'mcc',
        groupby: List[str] = None,
        verbose: bool = True, ):
    metrics = [metrics] if not isinstance(metrics, list) else list(metrics)
    groupby = ['n_components', 'cca_reg', 'clf_reg'] if groupby is None else groupby

    def _xtract(_df, _metric):
        _df = _df.loc[_df.metric == _metric]
        _df = _df.groupby(by=groupby, as_index=False).mean()
        _df = _df.iloc[_df.value.argmax()]
        _best = {key: _df[key] for key in groupby}
        return _best

    df = results.copy(deep=True)

    for m in metrics:
        best = _xtract(df, m)
        df = results.loc[
            (results.n_components == best['n_components']) &
            (results.cca_reg == best['cca_reg'])
        ]
        # TODO: this is not fully implemented yet

    for m in results.metric.unique():
        selected_df = results.loc[
            (results.n_components == best['n_components']) &
            (results.cca_reg == best['cca_reg']) &
            (results.clf_reg == best['clf_reg']) &
            (results.metric == m),
        ]
        best.update({m: selected_df.value.mean()})

    if verbose:
        msg = 'best hyperparams:\n\n'
        msg += 'n_components:\t{:d},\n'
        msg += 'cca_reg:\t{:.3e},\n'
        msg += 'clf_reg:\t{:.3e},\n'
        msg += 'pred_r:\t\t{:.4f}\n'
        msg += 'mcc:\t\t{:.4f}\n\n'
        msg = msg.format(
            best['n_components'],
            best['cca_reg'],
            best['clf_reg'],
            best['pred_r'],
            best['mcc'],
        )
        print(msg)

    return best


def extract_best_hyperparams(results, metric: str = None, verbose: bool = True):
    def _xtract(_df, criterion):
        _df = _df.loc[_df.metric == criterion]
        _df = _df.groupby(['n_components', 'cca_reg', 'clf_reg'], as_index=False).mean()
        _df = _df.iloc[_df.value.argmax()]
        _best = {
            'n_components': int(_df.n_components),
            'cca_reg': float(_df.cca_reg),
            'clf_reg': float(_df.clf_reg),
        }
        return _best

    if metric is None:
        best = _xtract(results, 'pred_r')
        df = results.loc[
            (results.n_components == best['n_components']) &
            (results.cca_reg == best['cca_reg'])
        ]
        best = _xtract(df, 'mcc')
    else:
        best = _xtract(results, metric)

    for m in ['pred_r', 'bal_acc', 'mcc']:
        selected_df = results.loc[
            (results.n_components == best['n_components']) &
            (results.cca_reg == best['cca_reg']) &
            (results.clf_reg == best['clf_reg']) &
            (results.metric == m),
        ]
        best.update({m: selected_df.value.mean()})

    if verbose:
        msg = 'best hyperparams:\n\n'
        msg += 'n_components:\t{:d},\n'
        msg += 'cca_reg:\t{:.3e},\n'
        msg += 'clf_reg:\t{:.3e},\n'
        msg += 'pred_r:\t\t{:.4f}\n'
        msg += 'mcc:\t\t{:.4f}\n\n'
        msg = msg.format(
            best['n_components'],
            best['cca_reg'],
            best['clf_reg'],
            best['pred_r'],
            best['mcc'],
        )
        print(msg)

    return best


def prepare_cca_data(
        h_load_file: str,
        min_nb_trials: int = -1,
        time_range: range = range(45, 46),
        target: bool = True,
        normalize_mode: str = 'none',
        full_align: bool = False,
        augment: bool = False,
        xv_folds: int = 5,
        which_fold: int = 0,
        verbose: bool = False, ):
    key_dff = 'target_dffs' if target else 'nontarget_dffs'
    key_label = 'target_labels' if target else 'nontarget_labels'
    key_freq = 'target_freqs' if target else 'nontarget_freqs'

    raw_data = load_target_nontarget(h_load_file)

    unique_labels = np.unique(list(raw_data[key_label].values())[0])
    unique_freqs = np.unique(list(raw_data[key_freq].values())[0])

    aligned_tst, aligned_trn = {}, {}
    aligned_label_tst, aligned_label_trn = {}, {}
    for name, dff in raw_data[key_dff].items():
        if dff.shape[1] < min_nb_trials:
            continue

        if verbose:
            msg = 'expt name: {:s},\t\t(nt, ntrials, nc) = ({:d}, {:d}, {:d})'
            msg = msg.format(name, *dff.shape)
            print(msg)

        label = raw_data[key_label][name]
        freq = raw_data[key_freq][name]

        local_tst, local_trn = {}, {}
        local_label_tst, local_label_trn = {}, {}
        for ll in unique_labels:
            if full_align:
                for ff in unique_freqs:
                    _idxs = np.where((label == ll) & (freq == ff))[0]
                    tst_indxs, trn_indxs = train_test_split(
                        n_samples=len(_idxs),
                        xv_folds=xv_folds,
                        which_fold=which_fold,
                    )
                    _key = 'll:{:d}-ff:{:d}'.format(ll, ff)
                    local_tst[_key] = dff[time_range][:, _idxs[tst_indxs], :]
                    local_trn[_key] = dff[time_range][:, _idxs[trn_indxs], :]
                    local_label_tst[_key] = label[_idxs[tst_indxs]]
                    local_label_trn[_key] = label[_idxs[trn_indxs]]

            else:
                _idxs = np.where(label == ll)[0]
                tst_indxs, trn_indxs = train_test_split(
                    n_samples=len(_idxs),
                    xv_folds=xv_folds,
                    which_fold=which_fold,
                )
                _key = 'll:{:d}'.format(ll)
                local_tst[_key] = dff[time_range][:, _idxs[tst_indxs], :]
                local_trn[_key] = dff[time_range][:, _idxs[trn_indxs], :]
                local_label_tst[_key] = label[_idxs[tst_indxs]]
                local_label_trn[_key] = label[_idxs[trn_indxs]]

        aligned_tst[name] = local_tst
        aligned_trn[name] = local_trn
        aligned_label_tst[name] = local_label_tst
        aligned_label_trn[name] = local_label_trn

    cat_tst, labels_tst = combine(
        data=aligned_tst,
        labels=aligned_label_tst,
        augment=augment,
        verbose=verbose,
    )
    cat_trn, labels_trn = combine(
        data=aligned_trn,
        labels=aligned_label_trn,
        augment=augment,
        verbose=verbose,
    )

    if normalize_mode == 'zscore':
        cat_tst = {name: zscore(v.reshape(-1, v.shape[-1]), axis=0).reshape(*v.shape) for name, v in cat_tst.items()}
        cat_trn = {name: zscore(v.reshape(-1, v.shape[-1]), axis=0).reshape(*v.shape) for name, v in cat_trn.items()}
        # cat_trn = {name: zscore(v, axis=0) for name, v in cat_trn.items()}
    elif normalize_mode == 'center':
        global_mean = np.mean(
            [item.mean() for item in cat_tst.values()] +
            [item.mean() for item in cat_trn.values()]
        )
        global_sd = np.mean(
            [item.std() for item in cat_tst.values()] +
            [item.std() for item in cat_trn.values()]
        )
        cat_tst = {name: (v - global_mean) / global_sd for name, v in cat_tst.items()}
        cat_trn = {name: (v - global_mean) / global_sd for name, v in cat_trn.items()}

    data_tst = {
        'aligned': aligned_tst,
        'cat': cat_tst,
        'lbl': labels_tst,
        'x': list(cat_tst.values()),
        'y': np.concatenate(list(labels_tst.values())),
    }
    data_trn = {
        'aligned': aligned_trn,
        'cat': cat_trn,
        'lbl': labels_trn,
        'x': list(cat_trn.values()),
        'y': np.concatenate(list(labels_trn.values())),
    }
    return data_tst, data_trn, list(cat_tst.keys())


def combine(
        data: Dict[str, dict],
        labels: Dict[str, dict],
        augment: bool = False,
        verbose: bool = False, ):
    num_samples = [v.shape[1] for d in data.values() for v in d.values()]
    min_num_samples_global = min(num_samples)
    max_num_samples_global = max(num_samples)

    aligned_num_samples = {name: [v.shape[1] for v in d.values()] for name, d in data.items()}
    num_samples_dict = {name: sum(v) for name, v in aligned_num_samples.items()}
    min_num_shared_samples = min(num_samples_dict.values())

    if augment:
        raise NotImplementedError
        # TODO: figure out augmentation

    else:
        data_cat = {name: np.concatenate(list(d.values()), axis=1) for name, d in data.items()}
        labels_cat = {name: np.concatenate(list(d.values())) for name, d in labels.items()}

        data_final = {name: d[:, :min_num_shared_samples, :] for name, d in data_cat.items()}
        labels_final = {name: d[:min_num_shared_samples] for name, d in labels_cat.items()}

    if verbose:
        print('\n\n')
        pprint(num_samples)
        print(min_num_samples_global, max_num_samples_global, min_num_shared_samples, '\n')
        pprint(list(list(data.values())[0].keys()))
        pprint(aligned_num_samples)
        # pprint(num_samples_dict)
        pprint({name: v.shape for name, v in data_cat.items()})

    return data_final, labels_final


def load_target_nontarget(h_load_file: str):
    target_dffs, nontarget_dffs = {}, {}
    target_labels, nontarget_labels = {}, {}
    target_freqs, nontarget_freqs = {}, {}

    f = h5py.File(h_load_file, 'r')
    for name in f:
        behavior = f[name]['behavior']
        passive = f[name]['passive']

        good_cells_b = np.array(behavior["good_cells"], dtype=int)
        good_cells_p = np.array(passive["good_cells"], dtype=int)
        good_cells = set(good_cells_b).intersection(set(good_cells_p))
        good_cells = sorted(list(good_cells))

        dff = np.array(behavior['dff'], dtype=float)[..., good_cells]

        trial_info = {}
        for k, v in behavior["trial_info"].items():
            trial_info[k] = np.array(v, dtype=int)

        target_indxs = np.where(trial_info['target'])[0]
        nontarget_indxs = np.where(trial_info['nontarget'])[0]

        valid_target_indxs = np.where(
            trial_info['hit'][target_indxs] +
            trial_info['miss'][target_indxs]
        )[0]
        valid_nontarget_indxs = np.where(
            trial_info['correctreject'][target_indxs] +
            trial_info['falsealarm'][target_indxs]
        )[0]

        target_indxs = target_indxs[valid_target_indxs]
        nontarget_indxs = nontarget_indxs[valid_nontarget_indxs]

        target_dffs[name] = dff[:, target_indxs, :]
        nontarget_dffs[name] = dff[:, nontarget_indxs, :]

        target_labels[name] = trial_info['hit'][target_indxs]
        nontarget_labels[name] = trial_info['correctreject'][nontarget_indxs]

        target_freqs[name] = trial_info['stimfrequency'][target_indxs]
        nontarget_freqs[name] = trial_info['stimfrequency'][nontarget_indxs]
    f.close()

    target_nontarget_data = {
        'target_dffs': target_dffs,
        'target_labels': target_labels,
        'target_freqs': target_freqs,
        'nontarget_dffs': nontarget_dffs,
        'nontarget_labels': nontarget_labels,
        'nontarget_freqs': nontarget_freqs,
    }
    return target_nontarget_data


# --------------------------- plotting functions -------------------------------
# ------------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, best: dict, figsize=(9, 7), dpi=70):
    sns.set_style('whitegrid')
    plt.figure(figsize=figsize, dpi=dpi)

    selected_df = results.loc[
        (results.cca_reg == best['cca_reg']) &
        (results.clf_reg == best['clf_reg'])
    ]
    sns.lineplot(data=selected_df, x='n_components', y='value', hue='metric',
                 style='metric', markers=True, dashes=False, lw=3, markersize=5)
    plt.axvline(best['n_components'], ls=':', color='k', alpha=0.7)
    plt.xlabel('n_components', fontsize=12)
    plt.xlabel('performance', fontsize=12)

    msg = '# CCA componens vs. mcc, balanced accuracy, and pred correlations\n'
    msg += 'performance increases as we introduce more and more components then saturates\n\n'
    msg += 'best # components: {:d},  predicted canonical correlations = {:.3f}\n'
    msg += 'best avg performance:  i) mcc = {:.3f},  ii) balanced accuracy = {:.3f}'
    msg = msg.format(best['n_components'], best['pred_r'], best['mcc'], best['bal_acc'])
    plt.suptitle(msg, fontsize=13, y=1.02)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_reg_comparison2(
        results: pd.DataFrame,
        best: dict,
        metric: str = 'mcc',
        components: List[int] = None,
        groupby: List[int] = None,
        scale: str = 'log',
        figsize=None,
        dpi=70, ):

    components = [75, 77, 78, 79, 80, 81] if components is None else components
    groupby = ['cca_reg', 'clf_reg'] if groupby is None else groupby
    assert len(groupby) == 2, "must provide 2 variabels to group by"

    sns.set_style('whitegrid')
    figsize = (2.25 * len(components), 6) if figsize is None else figsize
    fig, axes = plt.subplots(3, len(components), figsize=figsize, dpi=dpi)
    axes = axes.reshape(3, len(components))

    for idx, cc in enumerate(components):
        selected_df = results.loc[(results.n_components == cc) & (results.metric == metric)]
        df = selected_df.groupby(groupby, as_index=False).mean()
        df = df.pivot(index=groupby[0], columns=[groupby[1]], values='value')

        sns.heatmap(data=df, xticklabels=False, yticklabels=False, vmin=0, vmax=best[metric], ax=axes[0, idx])
        axes[0, idx].set_title('# components = {:d}'.format(cc))
        axes[0, idx].set_aspect('equal')

        for i in range(2):
            a = df.mean(i)
            axes[i + 1, idx].plot(a, lw=3)
            axes[i + 1, idx].axvline(a.index[a.argmax()], lw=2, ls='--', color='tomato')
            axes[i + 1, idx].set_xlabel(a.index.name)
            axes[i + 1, idx].set_xscale(scale)

    fig.tight_layout()
    plt.show()


def plot_reg_comparison(results, best, components: List[int] = None,
                        scale: str = 'log', figsize=None, dpi=70):

    components = [75, 77, 78, 79, 80, 81] if components is None else components
    figsize = (2.25 * len(components), 6) if figsize is None else figsize

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, len(components), figsize=figsize, dpi=dpi)
    axes = axes.reshape(3, len(components))

    for idx, cc in enumerate(components):
        selected_df = results.loc[(results.n_components == cc) & (results.metric == 'mcc')]
        df = selected_df.groupby(['cca_reg', 'clf_reg'], as_index=False).mean()
        df = df.pivot(index='cca_reg', columns=['clf_reg'], values='value')

        sns.heatmap(data=df, xticklabels=False, yticklabels=False, vmin=0, vmax=best['mcc'], ax=axes[0, idx])
        axes[0, idx].set_title('# components = {:d}'.format(cc))
        axes[0, idx].set_aspect('equal')

        for i in range(2):
            a = df.mean(i)
            axes[i + 1, idx].plot(a, lw=3)
            axes[i + 1, idx].axvline(a.index[a.argmax()], lw=2, ls='--', color='tomato')
            axes[i + 1, idx].set_xlabel(a.index.name)
            axes[i + 1, idx].set_xscale(scale)

    fig.tight_layout()
    plt.show()


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cm",
        help="a comment about this fit",
        type=str,
    )
    parser.add_argument(
        "-C",
        help="regularizer hyperparams",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--clf_type",
        help="classifier type, choices: {'logreg', 'svm'}",
        type=str,
        choices={'logreg', 'svm', 'mlp'},
        default='svm',
    )
    parser.add_argument(
        "--hidden_size",
        help="hidden size, only when using mlp",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--penalty",
        help="regularization type, choices: {'l1', 'l2'}",
        type=str,
        choices={'l1', 'l2'},
        default='l1',
    )
    parser.add_argument(
        "--nb_seeds",
        help="number of different seeds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--xv_fold",
        help="num cross-validation folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--solver",
        help="choices: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}",
        type=str,
        choices={'liblinear', 'lbfgs', 'auto'},
        default='auto',
    )
    parser.add_argument(
        "--tol",
        help="classifier tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max_iter",
        help="max iter",
        type=int,
        default=int(1e6),
    )
    parser.add_argument(
        "--nb_std",
        help="outlier removal threshold",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_to_pieces",
        help="if True, will save each fit then combine them using combine_fits() module",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="verbosity",
        action="store_true",
    )
    parser.add_argument(
        "--base_dir",
        help="base dir where project is saved",
        type=str,
        default='Documents/Kanold',
    )

    return parser.parse_args()


def main():
    args = _setup_args()
    if args.solver == 'auto':
        args.solver = 'liblinear' if args.clf_type in ['logreg', 'svm'] else 'lbfgs'

    base_dir = pjoin(os.environ['HOME'], args.base_dir)
    results_dir = pjoin(base_dir, 'results')
    processed_dir = pjoin(base_dir, 'python_processed')
    h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(args.nb_std))

    tasks = get_tasks()
    seeds = [np.power(2, i) for i in range(args.nb_seeds)]

    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
