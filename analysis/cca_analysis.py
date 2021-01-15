import sys
import h5py
import rcca
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

sys.path.append('..')
from utils.generic_utils import *
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit_cca_loop(h_load_file: str, n_seeds: int = 3, save_file: str = None, **kwargs):
    default_args = {
        'seeds': [int(2 ** i) for i in range(max(1, n_seeds))],
        'min_nb_trials': 100,
        'target': True,
        'global_normalize': True,
        'augment_data': False,
        'xv_folds': 5,
        'time_range': range(45, 46),
        'num_ccs': np.arange(5, 101, 5),
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
            output_trn, output_tst = prepare_cca_data(
                h_load_file=h_load_file,
                min_nb_trials=default_args['min_nb_trials'],
                time_range=default_args['time_range'],
                target=default_args['target'],
                global_normalize=default_args['global_normalize'],
                augment_data=default_args['augment_data'],
                xv_folds=default_args['xv_folds'],
                which_fold=fold,
                verbose=False,
            )
            train_list, y_trn = output_trn['processed'], output_trn['labels']
            test_list, y_tst = output_tst['processed'], output_tst['labels']

            for n_components in tqdm(default_args['num_ccs'], leave=False):
                for reg in tqdm(default_args['cca_regs'], leave=False):
                    cca = rcca.CCA(
                        kernelcca=True,
                        ktype='linear',
                        numCC=n_components,
                        reg=reg,
                        verbose=False,
                    )
                    cca.train([item / np.sqrt(n_components) for item in train_list])
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

    results = reset_df(results)
    best = extract_best_hyperparams(results, metric='mcc', verbose=True)

    save_file = './results_{}.df'.format(now()) if save_file is None else save_file
    results.to_pickle(save_file)

    return results, best, default_args


def extract_best_hyperparams(results, metric: str = 'mcc', verbose: bool = True):
    best_nc = -1
    best_cca_reg = -1
    best_clf_reg = -1
    best_score = -1

    selected_df = results.loc[results.metric == metric]

    for cca_reg in results.cca_reg.unique():
        for clf_reg in results.clf_reg.unique():
            for n_comps in results.n_components.unique():
                current_score = selected_df.loc[
                    (selected_df.cca_reg == cca_reg) &
                    (selected_df.clf_reg == clf_reg) &
                    (selected_df.n_components == n_comps)
                ].value.mean()

                if current_score >= best_score:
                    best_score = current_score
                    best_cca_reg = cca_reg
                    best_clf_reg = clf_reg
                    best_nc = n_comps

    if verbose:
        msg = 'best hyperparams:\n\n'
        msg += 'n_components:\t{:d},\ncca_reg:\t{:.3e},\nclf_reg:\t{:.3e},\nscore:\t\t{:.4f}\n\n'
        msg = msg.format(best_nc, best_cca_reg, best_clf_reg, best_score)
        print(msg)

    best = {
        'n_component': best_nc,
        'cca_reg': best_cca_reg,
        'best_clf_reg': best_clf_reg,
        'score': best_score,
    }
    return best


def prepare_cca_data(
        h_load_file: str,
        min_nb_trials: int = -1,
        time_range: range = range(45, 46),
        target: bool = True,
        global_normalize: bool = True,
        augment_data: bool = False,
        xv_folds: int = 5,
        which_fold: int = 0,
        verbose: bool = False, ):
    key_dff = 'target_dffs' if target else 'nontarget_dffs'
    key_label = 'target_labels' if target else 'target_labels'
    key_freq = 'target_freqs' if target else 'target_freqs'

    raw_data = _load_target_nontarget(h_load_file)

    unique_labels = np.unique(list(raw_data[key_label].values())[0])
    unique_freqs = np.unique(list(raw_data[key_freq].values())[0])

    trn, tst = {}, {}
    df_trn, df_tst = pd.DataFrame(), pd.DataFrame()

    global_sds, global_means = [], []
    for name, dff in raw_data[key_dff].items():
        if dff.shape[1] < min_nb_trials:
            continue
        label = raw_data[key_label][name]
        freq = raw_data[key_freq][name]

        if verbose:
            print(name, dff.shape)

        _trn, _tst = [], []
        _df_trn, _df_tst = pd.DataFrame(), pd.DataFrame()

        for ll in unique_labels:
            for ff in unique_freqs:
                _idxs = np.where((label == ll) & (freq == ff))[0]
                global_sds.append(np.std(dff[time_range]))
                global_means.append(np.std(dff[time_range]))

                tst_indxs, trn_indxs = train_test_split(len(_idxs), xv_folds=xv_folds, which_fold=which_fold)
                _tst.append(dff[time_range][:, tst_indxs, :])
                _trn.append(dff[time_range][:, trn_indxs, :])

                timepoint = np.expand_dims(time_range, 1)
                data_dict_tst = {
                    'timepoint': np.repeat(timepoint, len(tst_indxs), axis=1).flatten(),
                    'name': [name] * len(tst_indxs) * len(time_range),
                    'label': [ll] * len(tst_indxs) * len(time_range),
                    'freq': [ff] * len(tst_indxs) * len(time_range),
                }
                data_dict_trn = {
                    'timepoint': np.repeat(timepoint, len(trn_indxs), axis=1).flatten(),
                    'name': [name] * len(trn_indxs) * len(time_range),
                    'label': [ll] * len(trn_indxs) * len(time_range),
                    'freq': [ff] * len(trn_indxs) * len(time_range),
                }
                _df_tst = pd.concat([_df_tst, pd.DataFrame.from_dict(data_dict_tst)])
                _df_trn = pd.concat([_df_trn, pd.DataFrame.from_dict(data_dict_trn)])

        tst[name] = np.concatenate(_tst, axis=1)
        trn[name] = np.concatenate(_trn, axis=1)
        df_tst = pd.concat([df_tst, _df_tst])
        df_trn = pd.concat([df_trn, _df_trn])

    if global_normalize:
        global_sd = np.mean(global_sds)
        global_mean = np.mean(global_means)
        trn = {k: (v - global_mean) / global_sd for k, v in trn.items()}
        tst = {k: (v - global_mean) / global_sd for k, v in tst.items()}

    train_list, y_trn, test_list, y_tst = _get_xy(trn, tst, df_trn, df_tst, augment_data)
    output_trn = {'raw': trn, 'processed': train_list, 'labels': y_trn, 'df': df_trn}
    output_tst = {'raw': tst, 'processed': test_list, 'labels': y_tst, 'df': df_tst}
    return output_trn, output_tst


def _get_xy(trn, tst, df_trn, df_tst, augment_data: bool = False):
    if not augment_data:
        num_tst = min([item.shape[1] for item in tst.values()])
        num_trn = min([item.shape[1] for item in trn.values()])
        train_list = [item[:, :num_trn, :].reshape(-1, item.shape[-1]) for item in trn.values()]
        test_list = [item[:, :num_tst, :].reshape(-1, item.shape[-1]) for item in tst.values()]

        train_labels_list = [df_trn.loc[df_trn.name == name].label.tolist()[:num_trn] for name in trn.keys()]
        test_labels_list = [df_tst.loc[df_tst.name == name].label.tolist()[:num_tst] for name in tst.keys()]
        y_trn, y_tst = tuple(map(np.concatenate, [train_labels_list, test_labels_list]))

    else:
        raise NotImplementedError
    # TODO: figure out augmentation

    return train_list, y_trn, test_list, y_tst


def _load_target_nontarget(h_load_file: str):
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

        target_dffs[name] = dff[:, target_indxs, :]
        nontarget_dffs[name] = dff[:, nontarget_indxs, :]

        target_labels[name] = trial_info['hit'][target_indxs]
        nontarget_labels[name] = trial_info['correctreject'][nontarget_indxs]

        target_freqs[name] = trial_info['stimfrequency'][target_indxs]
        nontarget_freqs[name] = trial_info['stimfrequency'][nontarget_indxs]
    f.close()

    raw_data = {
        'target_dffs': target_dffs,
        'target_labels': target_labels,
        'target_freqs': target_freqs,
        'nontarget_dffs': nontarget_dffs,
        'nontarget_labels': nontarget_labels,
        'nontarget_freqs': nontarget_freqs,
    }
    return raw_data


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
