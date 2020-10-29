import os
import h5py
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import norm
from typing import Union, List
from copy import deepcopy as dc
from os.path import join as pjoin
from collections import namedtuple

from sklearn.metrics import matthews_corrcoef
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.generic_utils import merge_dicts, save_obj, now, reset_df

LDA = namedtuple('LDA', ('name', 'X', 'Y', 'trajs', 'clfs'))


def run_lda_analysis(
        cm: str,
        load_file: str,
        results_dir: str,
        trial_types: List[str],
        shrinkage: Union[float, str] = 'auto',
        xv_fold: int = 5,
        random_state: int = 42,
        verbose: bool = True,
):

    random.seed(random_state)
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)

    lbl2idx = {lbl: i for (i, lbl) in enumerate(trial_types)}
    idx2lbl = {i: k for k, i in lbl2idx.items()}

    msg = "[INFO] running LDA analysis using shrinkage = '{}'\n"
    msg += "[INFO] cm: {}, class labels: {}\n"
    msg = msg.format(shrinkage, cm, trial_types)
    if verbose:
        print(msg)

    # save dir
    save_dir = pjoin(results_dir, 'lda', cm)
    os.makedirs(save_dir, exist_ok=True)

    fit_metadata = {
        'shrinkage': shrinkage,
        'lbl2idx': lbl2idx,
        'idx2lbl': idx2lbl,
        'save_dir': save_dir,
        'datetime': now(),
    }
    save_obj(fit_metadata, 'fit_metadata.npy', save_dir, 'np')

    for shuffle_labels in [False, True]:
        for dim in [1, 2, 3]:
            results, lda_dict = _lda(load_file, shrinkage, dim, xv_fold, lbl2idx, idx2lbl, rng, shuffle_labels, verbose)

            # save
            file_name = 'results_{:d}d_shuffled.df' if shuffle_labels else 'results_{:d}d.df'
            save_obj(results, file_name.format(dim), save_dir, 'df', verbose)
            file_name = 'extras_{:d}d_shuffled.pkl' if shuffle_labels else 'extras_{:d}d.pkl'
            save_obj(lda_dict, file_name.format(dim), save_dir, 'pkl', verbose)


def _lda(load_file, shrinkage, dim, xv_fold, lbl2idx, idx2lbl, rng, shuffle_labels, verbose):
    lda_dict = {}
    results_dictlist = []
    h5_file = h5py.File(load_file, "r")
    pbar = tqdm(h5_file, dynamic_ncols=True, disable=not verbose)
    for name in pbar:
        msg = "shuffled, {:d}d, {}" if shuffle_labels else "{:d}d, {}"
        pbar.set_description(msg.format(dim, name))
        behavior = h5_file[name]["behavior"]
        trial_info_grp = behavior["trial_info"]

        good_cells = np.array(behavior["good_cells"], dtype=int)
        dff = np.array(behavior["dff"], dtype=float)[..., good_cells]
        nt, ntrials, _ = dff.shape

        trial_info = {}
        for k, v in trial_info_grp.items():
            trial_info[k] = np.array(v, dtype=int)

        if not set(lbl2idx.keys()).issubset(set(trial_info.keys())):
            if verbose:
                print("missing some trial types, skipping {} . . . ".format(name))
            continue

        lbls = []
        dff_combined = []
        for trial in lbl2idx:
            cond = trial_info[trial] == 1
            lbls.extend([trial] * cond.sum())
            dff_combined.append(dff[:, cond, :])

        y = np.array([lbl2idx[k] for k in lbls])
        dff_combined = np.concatenate(dff_combined, axis=1)

        vld_indxs = []
        for i in idx2lbl:
            idxs = np.where(y == i)[0]
            nb_vld = int(np.ceil(len(idxs) / xv_fold))
            vld_indxs.extend(random.sample(list(idxs), nb_vld))
        trn_indxs = np.delete(range(len(y)), vld_indxs)
        assert set(trn_indxs).isdisjoint(set(vld_indxs))

        y_trn, y_vld = y[trn_indxs], y[vld_indxs]

        num_samples = np.array([len(np.where(y_trn == i)[0]) for i in idx2lbl.keys()])
        if any(num_samples < 2):
            if verbose:
                print("not enough samples, skipping {} . . .".format(name))
            continue

        performance = np.zeros(nt)
        embedded = np.zeros((nt, len(vld_indxs), dim))

        _clfs = {}
        for t in tqdm(range(nt), leave=False, disable=not verbose):
            x_trn, x_vld = dff_combined[t][trn_indxs], dff_combined[t][vld_indxs]
            if shuffle_labels:
                while True:
                    y_shuffled = dc(y_trn)
                    rng.shuffle(y_shuffled)
                    if not np.all(y_shuffled == y_trn):
                        break
                y_trn = y_shuffled
            clf = LinearDiscriminantAnalysis(
                n_components=dim,
                solver='eigen',
                shrinkage=shrinkage,
            ).fit(x_trn, y_trn)
            z = clf.transform(x_vld)
            embedded[t] = z
            _clfs[t] = clf

            y_pred = clf.predict(x_vld)
            performance[t] = matthews_corrcoef(y_vld, y_pred)

        embedded_dict = {lbl: embedded[:, y_vld == idx, :] for lbl, idx in lbl2idx.items()}

        mu0 = embedded.mean(1)
        mu_dict = {lbl: z.mean(1) for lbl, z in embedded_dict.items()}
        scatter_between = {
            lbl: z.shape[1] * norm(
                mu_dict[lbl] - mu0,
                axis=-1,
                keepdims=True,
            )
            for lbl, z in embedded_dict.items()
        }
        scatter_within = {
            lbl: z.shape[1] * np.concatenate(
                tuple(
                    norm(
                        z[:, i, :] - mu_dict[lbl],
                        axis=-1,
                        keepdims=True,
                    )
                    for i in range(z.shape[1])
                ), axis=-1,
            ).mean(-1, keepdims=True)
            for lbl, z in embedded_dict.items()
        }
        sb = np.concatenate(list(scatter_between.values()), axis=-1).sum(-1)
        sw = np.concatenate(list(scatter_within.values()), axis=-1).sum(-1)

        com_distances_dict = {
            lbl: np.concatenate(
                tuple(
                    norm(
                        mu - mu_prime,
                        axis=-1,
                        keepdims=True,
                    )
                    for mu_prime in mu_dict.values()
                ), axis=-1,
            ).sum(-1)
            for lbl, mu in mu_dict.items()
        }
        d = np.concatenate(
            list(
                np.expand_dims(item, axis=-1)
                for item in com_distances_dict.values()
            ), axis=-1,
        ).sum(-1)

        data_dict = {
            'name': [name] * nt,
            'timepoint': range(nt),
            'performance': performance,
            'distance': d,
            'sb': sb,
            'sw': sw,
            'J': sb / np.maximum(sw, 1e-8),
        }
        results_dictlist.append(data_dict)
        lda_dict[name] = LDA(name, dff_combined, y, embedded_dict, _clfs)

    # merge all results together, can be used to get df
    results = merge_dicts(results_dictlist)
    results = pd.DataFrame.from_dict(results)
    results = _compute_best_t(results)

    return results, lda_dict


def _compute_best_t(results: pd.DataFrame):
    names = results.name.unique().tolist()
    results['best_t'] = -1
    for name in names:
        df = results.loc[results.name == name]
        grouped = df.groupby(['timepoint']).mean()

        mean_j = grouped.J.to_numpy()
        mean_d = grouped.distance.to_numpy()
        mean_p = grouped.performance.to_numpy()

        mean_j /= np.linalg.norm(mean_j)
        mean_d /= np.linalg.norm(mean_d)
        mean_p /= np.linalg.norm(mean_p)

        v = (mean_j + mean_d + mean_p) / 3
        best_t = np.argmax(v)
        results.loc[results.name == name, 'best_t'] = best_t

    grouped = results.groupby(['timepoint']).mean()

    mean_j = grouped.J.to_numpy()
    mean_d = grouped.distance.to_numpy()
    mean_p = grouped.performance.to_numpy()

    mean_j /= np.linalg.norm(mean_j)
    mean_d /= np.linalg.norm(mean_d)
    mean_p /= np.linalg.norm(mean_p)

    v = (mean_j + mean_d + mean_p) / 3
    best_t_global = np.argmax(v)
    results['best_t_global'] = best_t_global

    return reset_df(results)


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_std",
        help="outlier removal threshold",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--xv_fold",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        help="random seed",
        type=int,
        default=42,
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
        default='Documents/PROJECTS/Kanold',
    )

    return parser.parse_args()


def main():
    args = _setup_args()

    base_dir = pjoin(os.environ['HOME'], args.base_dir)
    results_dir = pjoin(base_dir, 'results')
    h_load_file = pjoin(base_dir, 'python_processed', "organized_nb_std={:d}.h5".format(args.nb_std))

    runs = {
        '4way': ['hit', 'miss', 'correctreject', 'falsealarm'],
        'stimfreq': ['target7k', 'target10k', 'nontarget14k', 'nontarget20k'],
    }

    for cm, trial_types in runs.items():
        run_lda_analysis(
            cm=cm,
            load_file=h_load_file,
            results_dir=results_dir,
            shrinkage='auto',
            trial_types=trial_types,
            xv_fold=args.xv_fold,
            random_state=args.seed,
            verbose=args.verbose,
        )

    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
