import shutil
import joblib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from operator import methodcaller
from collections import defaultdict
from os.path import join as pjoin
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm


def get_tasks():
    raw_tasks = ['hit', 'miss', 'correctreject', 'falsealarm']
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


def reset_df(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    df = df.apply(pd.to_numeric, downcast="integer", errors="ignore")
    return df


def save_obj(data: Any, file_name: str, save_dir: str, mode: str = 'np', verbose: bool = True):
    _allowed_modes = ['np', 'df', 'pkl', 'joblib']
    with open(pjoin(save_dir, file_name), 'wb') as f:
        if mode == 'np':
            np.save(f.name, data)
        elif mode == 'df':
            pd.to_pickle(data, f)
        elif mode == 'pkl':
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif mode == 'joblib':
            joblib.dump(data, f)
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


def now(exclude_hour_min: bool = False):
    if exclude_hour_min:
        return datetime.now().strftime("[%Y_%m_%d]")
    else:
        return datetime.now().strftime("[%Y_%m_%d_%H-%M]")
