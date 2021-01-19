import os
import re
import shutil
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from operator import methodcaller
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import normalize
from os.path import join as pjoin
from datetime import datetime
from scipy import spatial
from tqdm import tqdm


def orthonormalize(u: np.ndarray, orthogonal: bool = True, normal: bool = True):
    x = u.copy()
    assert x.shape[1] >= 2, "must provide more than 2 vectors"

    if orthogonal:
        a = x[:, 0]
        b = x[:, 1].copy()
        b -= (np.dot(a, b) / np.dot(a, a)) * a

        if x.shape[1] == 2:
            x_orth = np.array([a, b]).T
        elif x.shape[1] == 3:
            c = x[:, 2].copy()
            c -= (np.dot(a, c) / np.dot(a, a)) * a
            c -= (np.dot(b, c) / np.dot(b, b)) * b
            x_orth = np.array([a, b, c]).T
        else:
            raise NotImplementedError
        x = x_orth

    if normal:
        x = normalize(x, axis=0)

    return x


def cos_similarity(i, ii):
    return 1 - spatial.distance.cosine(i, ii)


def train_test_split(n_samples: int, xv_folds: int = 5, which_fold: int = 0):
    which_fold = which_fold % xv_folds
    num_tst_indxs = int(np.ceil(n_samples / xv_folds))

    start = which_fold * num_tst_indxs
    stop = min((which_fold + 1) * num_tst_indxs, n_samples)
    tst_indxs = range(start, stop)
    trn_indxs = np.delete(range(n_samples), tst_indxs)

    assert not set(tst_indxs).intersection(set(trn_indxs))
    return tst_indxs, trn_indxs


def load_dfs(load_dir: str) -> Dict[str, pd.DataFrame]:
    def _get_file(file_list, pattern):
        return next(filter(re.compile(pattern).match, file_list), None)

    flist = os.listdir(load_dir)
    coeffs = pd.read_pickle(pjoin(load_dir, _get_file(flist, r'coeffs_\[')))
    performances = pd.read_pickle(pjoin(load_dir, _get_file(flist, r'performances_\[')))
    coeffs_filtered = pd.read_pickle(pjoin(load_dir, _get_file(flist, r'coeffs_filtered_\[')))
    performances_filtered = pd.read_pickle(pjoin(load_dir, _get_file(flist, r'performances_filtered_\[')))

    df_all = {
        'performances': performances,
        'performances_filtered': performances_filtered,
        'coeffs': coeffs,
        'coeffs_filtered': coeffs_filtered,
    }
    return df_all


def smoothen(arr: np.ndarray, filter_sz: int = 5):
    shape = arr.shape
    assert 1 <= len(shape) <= 2, "1 <= dim <= 2d"

    kernel = np.ones(filter_sz) / filter_sz
    if len(shape) == 1:
        return np.convolve(arr, kernel, mode='same')
    else:
        smoothed = np.zeros(arr.shape)
        for i in range(arr.shape[0]):
            smoothed[i] = np.convolve(arr[i], kernel, mode='same')
        return smoothed


def downsample(data, xy, xbins, ybins, normal=True):
    xbins = sorted(xbins)
    ybins = sorted(ybins)
    assert len(xbins) == len(ybins)
    nbins = len(xbins)

    delta_x = (max(xbins) - min(xbins)) / nbins
    delta_y = (max(ybins) - min(ybins)) / nbins

    nc = xy.shape[0]
    downsampled = np.zeros((nbins, nbins))
    _norm = np.zeros((nbins, nbins))
    for cell in range(nc):
        x, y = xy[cell]

        if x == max(xbins):
            bin_i = -1
        else:
            bin_i = int(np.floor(
                (x - min(xbins)) / delta_x
            ))
        if y == max(ybins):
            bin_j = -1
        else:
            bin_j = int(np.floor(
                (y - min(ybins)) / delta_y
            ))

        downsampled[bin_j, bin_i] += data[cell]
        _norm[bin_j, bin_i] += 1

    if normal:
        return np.divide(downsampled, np.maximum(_norm, 1e-8))
    else:
        return downsampled


def get_dirs(nb_std: int = 1, base_dir: str = 'Documents/A1'):
    base_dir = pjoin(os.environ['HOME'], base_dir)
    data_dir = pjoin(base_dir, 'Data')
    results_dir = pjoin(base_dir, 'results')
    processed_dir = pjoin(base_dir, 'python_processed')
    h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(nb_std))
    df_load_file = pjoin(processed_dir, 'processed_nb_std={:d}'.format(nb_std), "all.df")

    dirs_dict = {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'results_dir': results_dir,
        'processed_dir': processed_dir,
        'h_load_file': h_load_file,
        'df_load_file': df_load_file,
    }
    return dirs_dict


def get_tasks(include_passive: bool = False):
    raw_tasks = ['hit', 'miss', 'correctreject', 'falsealarm']
    if include_passive:
        raw_tasks.append('passive')
    tasks = []
    for i, t1 in enumerate(raw_tasks):
        reduced = [item for item in raw_tasks[i:] if item != t1]
        for j, t2 in enumerate(reduced):
            tasks.append('{:s}/{:s}'.format(t1, t2))
    tasks += [
        'target7k/nontarget14k', 'target7k/nontarget20k',
        'target10k/nontarget14k', 'target10k/nontarget20k',
    ]
    return tasks


def reset_df(df: pd.DataFrame, downcast: str = "integer") -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    if downcast != 'none':
        df = df.apply(pd.to_numeric, downcast=downcast, errors="ignore")
    return df


def save_obj(obj: Any, file_name: str, save_dir: str = '.', mode: str = 'np', verbose: bool = True):
    _allowed_modes = ['np', 'df', 'pkl', 'joblib']
    with open(pjoin(save_dir, file_name), 'wb') as f:
        if mode == 'np':
            np.save(f.name, obj)
        elif mode == 'df':
            pd.to_pickle(obj, f)
        elif mode == 'pkl':
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif mode == 'joblib':
            joblib.dump(obj, f)
        else:
            raise RuntimeError("invalid mode encountered, available options: {}".format(_allowed_modes))
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))


def merge_dicts(dict_list: List[dict], verbose: bool = True) -> Dict[str, list]:
    merged = defaultdict(list)
    dict_items = map(methodcaller('items'), dict_list)
    for k, v in tqdm(chain.from_iterable(dict_items), disable=not verbose, leave=False, desc="...merging dicts"):
        merged[k].extend(v)
    return dict(merged)


def rm_dirs(base_dir: str, dirs: List[str], verbose: bool = True):
    for x in dirs:
        dirpath = Path(base_dir, x)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
    if verbose:
        print("[PROGRESS] removed {} folders at {:s}".format(dirs, base_dir))


def now(exclude_hour_min: bool = True):
    if exclude_hour_min:
        return datetime.now().strftime("(%Y_%m_%d)")
    else:
        return datetime.now().strftime("(%Y_%m_%d_%H-%M)")


def isfloat(string: str):
    try:
        float(string)
        return True
    except ValueError:
        return False
