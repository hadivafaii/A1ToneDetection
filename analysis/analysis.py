import os
import sys
import h5py
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm.notebook import tqdm
from copy import deepcopy
from collections import defaultdict
from itertools import chain
from operator import methodcaller
from typing import List, Tuple, Dict
from datetime import datetime
from os.path import join as pjoin

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, make_scorer

sys.path.append('..')
from utils.generic_utils import now


def run_classification_analysis(
        load_file: str,
        cm: str,
        save_results_dir: str,
        tasks: List[str] = None,
        seeds: List[int] = 42,
        xv_fold: int = 5,
        save_to_pieces: bool = False,
        verbose: bool = True,
        **kwargs: Dict[str, str],) -> Tuple[pd.DataFrame]:

    _allowed_clf_types = ['logreg', 'svm']

    if not isinstance(seeds, list):
        seeds = [seeds]

    if tasks is None:
        tasks = [
            'hit/miss', 'hit/falsealarm', 'hit/correctreject',
            'correctreject/falsealarm', 'miss/falsealarm', 'target/nontarget',
        ]

    classifier_args = {
        'clf_type': 'logreg',
        'penalty': 'l1',
        'C': [1.000, 0.500, 0.100, 0.050, 0.010, 0.005],
        'tol': 1e-4,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'max_iter': 1000,
    }
    for k in classifier_args:
        if k in kwargs:
            classifier_args[k] = kwargs[k]

    if verbose:
        msg = "[INFO] running analysis using: {:d}-fold xv, {:d} different seeds.\n"
        msg += "[INFO] classifier options:\n\t{}\n"
        msg += "[INFO] option save_to_pieces is: {}"
        msg = msg.format(xv_fold, len(seeds), classifier_args, save_to_pieces)
        print(msg)

    logger = _setup_logger(classifier_args['clf_type'])
    save_dir, coeffs_dir, performances_dir, classifiers_dir = _mk_save_dirs(
        cm, save_results_dir, classifier_args, verbose)

    coeffs_dict_list = []
    performances_dict_list = []
    _classifiers = defaultdict(tuple)
    counter = 0

    h5_file = h5py.File(load_file, "r")
    pbar = tqdm(h5_file, disable=not verbose)
    for expt in pbar:
        behavior = h5_file[expt]["behavior"]
        trial_info_grp = behavior["trial_info"]

        good_cells = np.array(behavior["good_cells"], dtype=int)
        xy = np.array(behavior["xy"], dtype=float)[good_cells]
        dff = np.array(behavior["dff"], dtype=float)[..., good_cells]
        nt, _, nc = dff.shape

        trial_info = {}
        for k, v in trial_info_grp.items():
            trial_info[k] = np.array(v, dtype=int)

        for random_state in tqdm(seeds, leave=False, disable=not verbose):
            random.seed(random_state)
            np.random.seed(random_state)

            for reg_c in tqdm(classifier_args['C'], leave=False, disable=not verbose):
                for task in tqdm(tasks, leave=False, disable=not verbose):
                    try:
                        pos_lbl, neg_lbl = task.split('/')
                        pos = trial_info[pos_lbl]
                        neg = trial_info[neg_lbl]
                    except KeyError:
                        msg = 'missing trials, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, reg_c, task)
                        logger.info(msg)
                        continue

                    include_trials = np.logical_or(pos, neg)
                    pos = pos[include_trials]
                    neg = neg[include_trials]

                    nb_pos_samples = sum(pos)
                    nb_neg_samples = sum(neg)

                    if nb_pos_samples == 0 or nb_neg_samples == 0:
                        msg = 'no samples found, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, reg_c, task)
                        logger.info(msg)
                        continue

                    assert np.all(np.array([pos, neg]).T.sum(-1) == 1), \
                        "pos and neg labels should be mutually exclusive"

                    # get xv indices
                    pos_vld_indxs = random.sample(range(nb_pos_samples), int(np.ceil(nb_pos_samples / xv_fold)))
                    neg_vld_indxs = random.sample(range(nb_neg_samples), int(np.ceil(nb_neg_samples / xv_fold)))

                    pos_vld_indxs = np.where(pos)[0][pos_vld_indxs]
                    neg_vld_indxs = np.where(neg)[0][neg_vld_indxs]

                    vld_indxs = list(pos_vld_indxs) + list(neg_vld_indxs)
                    trn_indxs = list(set(range(nb_pos_samples + nb_neg_samples)).difference(set(vld_indxs)))

                    x = dff[:, include_trials, :]

                    mcc_all = np.zeros(nt)
                    accuracy_all = np.zeros(nt)
                    f1_all = np.zeros(nt)
                    confidence_all = np.zeros(nt)

                    for time_point in tqdm(range(nt), leave=False, disable=not verbose):
                        counter += 1
                        x_trn, x_vld = x[time_point][trn_indxs], x[time_point][vld_indxs]
                        y_trn, y_vld = pos[trn_indxs], pos[vld_indxs]

                        try:
                            if classifier_args['clf_type'] == 'logreg':
                                clf = LogisticRegression(
                                    penalty=classifier_args['penalty'],
                                    C=reg_c,
                                    tol=classifier_args['tol'],
                                    solver=classifier_args['solver'],
                                    class_weight=classifier_args['class_weight'],
                                    max_iter=classifier_args['max_iter'],
                                    random_state=random_state,
                                ).fit(x_trn, y_trn)
                            elif classifier_args['clf_type'] == 'svm':
                                clf = LinearSVC(
                                    penalty=classifier_args['penalty'],
                                    C=reg_c,
                                    tol=classifier_args['tol'],
                                    class_weight=classifier_args['class_weight'],
                                    dual=False,
                                    max_iter=classifier_args['max_iter'],
                                    random_state=random_state,
                                ).fit(x_trn, y_trn)
                            else:
                                msg = "invalid classifier type encountered: {:s}, valid options are: {}"
                                msg = msg.format(classifier_args['clf_type'], _allowed_clf_types)
                                raise ValueError(msg)

                            y_pred = clf.predict(x_vld)
                            k = "{}^{}^{}^{}^{}".format(expt, task, random_state, reg_c, time_point)
                            if save_to_pieces:
                                _save = pjoin(classifiers_dir, '{:09d}.npy'.format(counter))
                                np.save(_save, {k: (clf, x_vld, y_vld)})
                            else:
                                _classifiers[k] = (clf, x_vld, y_vld)

                        except ValueError:
                            msg = 'num trials too small, name = {:s}, seed = {:d}, C = {}, task = {}, t = {}'
                            msg = msg.format(expt, random_state, reg_c, task, time_point)
                            logger.info(msg)
                            break

                        mcc_all[time_point] = matthews_corrcoef(y_vld, y_pred)
                        accuracy_all[time_point] = accuracy_score(y_vld, y_pred)
                        f1_all[time_point] = f1_score(y_vld, y_pred)

                        confidence = clf.decision_function(x_vld)
                        confidence_all[time_point] = sum(abs(confidence[y_vld == y_pred]))

                        data_dict = {
                            'name': [expt] * nc,
                            'seed': [random_state] * nc,
                            'task': [task] * nc,
                            'reg_C': [reg_c] * nc,
                            'timepoint': [time_point] * nc,
                            'cell_indx': range(nc),
                            'coeffs': clf.coef_.squeeze(),
                            'x': xy[:, 0],
                            'y': xy[:, 1],
                        }
                        if save_to_pieces:
                            _save = pjoin(coeffs_dir, '{:09d}.npy'.format(counter))
                            np.save(_save, data_dict)
                        else:
                            coeffs_dict_list.append(data_dict)

                        msg = "name: {},  seed: {},  C: {},  task: {},  t: {}, "
                        msg = msg.format(expt, random_state, reg_c, task, time_point)
                        pbar.set_description(msg)

                    data_dict = {
                        'name': [expt] * nt * 4,
                        'seed': [random_state] * nt * 4,
                        'task': [task] * nt * 4,
                        'reg_C': [reg_c] * nt * 4,
                        'timepoint': np.tile(range(nt), 4),
                        'metric': ['mcc'] * nt + ['accuracy'] * nt + ['f1'] * nt + ['confidence'] * nt,
                        'score': np.concatenate([mcc_all, accuracy_all, f1_all, confidence_all]),
                    }
                    if save_to_pieces:
                        _save = pjoin(performances_dir, '{:09d}.npy'.format(counter))
                        np.save(_save, data_dict)
                    else:
                        performances_dict_list.append(data_dict)

    h5_file.close()

    fit_metadata = {
        'classifier_args': classifier_args,
        'save_dir': save_dir,
        'coeffs_dir': coeffs_dir,
        'performances_dir': performances_dir,
        'classifiers_dir': classifiers_dir,
        'datetime': now(),
    }
    np.save(pjoin(save_dir, 'fit_metadata.npy'), fit_metadata)

    if not save_to_pieces:
        _coeffs = pd.DataFrame.from_dict(_merg_dicts(coeffs_dict_list))
        _performances = pd.DataFrame.from_dict(_merg_dicts(performances_dict_list))

        # reset dfs
        _coeffs = reset_df(_coeffs)
        _performances = reset_df(_performances)

        # save
        file_name = "_coeffs.df"
        _coeffs.to_pickle(pjoin(fit_metadata['save_dir'], file_name))
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

        file_name = "_performances.df"
        _performances.to_pickle(pjoin(fit_metadata['save_dir'], file_name))
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

        file_name = "_classifiers.npy"
        np.save(pjoin(fit_metadata['save_dir'], file_name), _classifiers)
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

    return fit_metadata


def combine_results(run_dir: str, verbose: bool = True) -> Tuple[pd.DataFrame]:
    df_coeffs = pd.DataFrame()
    df_performances = pd.DataFrame()
    clf_dict = {}

    runs = sorted(os.listdir(run_dir))
    for x in tqdm(runs, '[PROGRESS] combining previous fits together', disable=not verbose):
        load_dir = pjoin(run_dir, x)
        _coeffs = pd.read_pickle(pjoin(load_dir, '_coeffs.df'))
        _performances = pd.read_pickle(pjoin(load_dir, '_performances.df'))
        _classifiers = np.load(pjoin(load_dir, '_classifiers.npy'), allow_pickle=True).item()

        assert not set(_classifiers.keys()).intersection(set(clf_dict.keys())),\
            "must have non-overlapping keys by design"

        df_coeffs = df_coeffs.append(_coeffs)
        df_performances = df_performances.append(_performances)
        clf_dict.update(_classifiers)

        # reg_cs = list(map(lambda s: float(s), filter(None, x.split(','))))

    df_coeffs = reset_df(df_coeffs)
    df_performances = reset_df(df_performances)

    output = _porocess_results(df_performances, df_coeffs, verbose)
    df_performances, df_performances_filtered, df_coeffs_filtered = output
    df_coeffs_filtered = _compute_feature_importances(df_coeffs_filtered, clf_dict)

    # save
    time_now = now()
    file_name = "coeffs_{:s}.df".format(time_now)
    df_coeffs.to_pickle(pjoin(run_dir, file_name))
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, run_dir))

    file_name = "coeffs_filtered_{:s}.df".format(time_now)
    df_coeffs_filtered.to_pickle(pjoin(run_dir, file_name))
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, run_dir))

    file_name = "performances_{:s}.df".format(time_now)
    df_performances.to_pickle(pjoin(run_dir, file_name))
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, run_dir))

    file_name = "performances_filtered_{:s}.df".format(time_now)
    df_performances_filtered.to_pickle(pjoin(run_dir, file_name))
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, run_dir))

    file_name = "classifiers_{:s}.df".format(time_now)
    np.save(pjoin(run_dir, file_name), clf_dict)
    if verbose:
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, run_dir))

    return df_coeffs, df_coeffs_filtered, df_performances, df_performances_filtered


def combine_fits(fit_metadata: Dict[str, str], verbose: bool = True):
    files = ['_coeffs.df', '_performances.df', '_classifiers.npy']
    listdir = os.listdir(fit_metadata['save_dir'])
    cond = set(files).issubset(set(listdir))

    if not cond:
        # coeffs
        dictdata_list = []
        dirs = sorted(os.listdir(fit_metadata['coeffs_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _coeffs together', disable=not verbose):
            load = pjoin(fit_metadata['coeffs_dir'], x)
            data_dict = np.load(load, allow_pickle=True).item()
            dictdata_list.append(data_dict)
        _coeffs = pd.DataFrame.from_dict(dictdata_list)

        # performances
        dictdata_list = []
        dirs = sorted(os.listdir(fit_metadata['performances_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _performances together', disable=not verbose):
            load = pjoin(fit_metadata['performances_dir'], x)
            data_dict = np.load(load, allow_pickle=True).item()
            dictdata_list.append(data_dict)
        _performances = pd.DataFrame.from_dict(dictdata_list)

        # classifiers
        _classifiers = defaultdict(tuple)
        dirs = sorted(os.listdir(fit_metadata['classifiers_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _classifiers together', disable=not verbose):
            load = pjoin(fit_metadata['classifiers_dir'], x)
            data_dict = np.load(load, allow_pickle=True).item()
            for k, v in data_dict.items():
                _classifiers[k] += v

        # reset dfs
        _coeffs = reset_df(_coeffs)
        _performances = reset_df(_performances)

        # save
        file_name = "_coeffs.df"
        _coeffs.to_pickle(pjoin(fit_metadata['save_dir'], file_name))
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

        file_name = "_performances.df"
        _performances.to_pickle(pjoin(fit_metadata['save_dir'], file_name))
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

        file_name = "_classifiers.npy"
        np.save(pjoin(fit_metadata['save_dir'], file_name), _classifiers)
        if verbose:
            print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, fit_metadata['save_dir']))

    else:
        if verbose:
            print('[PROGRESS] skipped combining, files found: {}'.format(files))


def _compute_feature_importances(df: pd.DataFrame, classifiers: dict, verbose: bool = True) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=['importances'], index=df.index)
    new_df = pd.concat([df, empty_df], axis=1)

    for k, (clf, x_vld, y_vld) in tqdm(classifiers.items(),
                                       desc='[PROGRESS] computing feature importances', disable=not verbose):
        name, task, random_state, reg_c, time_point = k.split('^')
        random_state, time_point = int(random_state), int(time_point)
        reg_c = float(reg_c)

        cond = (new_df.name == name) & \
               (new_df.task == task) & \
               (new_df.seed == random_state)

        _c = new_df.loc[cond, 'reg_C'].unique()
        _t = new_df.loc[cond, 'timepoint'].unique()
        assert len(_c) == len(_t) == 1, "filtered df must have only one unique selected timepoint or reg per expt/task"

        if not (_c[0] == reg_c and _t[0] == time_point):
            continue

        importance_result = permutation_importance(
            estimator=clf,
            X=x_vld,
            y=y_vld,
            n_repeats=10,
            scoring=make_scorer(matthews_corrcoef),
            random_state=random_state,
        )

        new_df.loc[cond, 'importances'] = pd.Series(
            importance_result['importances_mean'], index=new_df.loc[cond].index)

    return reset_df(new_df)


def reset_df(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    df = df.apply(pd.to_numeric, downcast="integer", errors="ignore")
    return df


def _porocess_results(df_performances: pd.DataFrame, df_coeffs: pd.DataFrame, verbose: bool = True) -> tuple:
    output = ()
    df_performances = _normalize_confidence_score(df_performances, verbose)
    df_performances = _detect_best_reg_timepoint(df_performances, verbose)
    output += (df_performances,)
    filtered_dfs = _filter_df([df_performances, df_coeffs], df_performances, verbose)
    output += tuple(filtered_dfs)
    return output


def _filter_df(dfs: List[pd.DataFrame], augmented_df: pd.DataFrame, verbose: bool = True) -> List[pd.DataFrame]:
    filtered_dfs = []
    for df in tqdm(dfs, desc='[PROGRESS] filtering dfs using best reg/timepoints', disable=not verbose):
        filtered_df = deepcopy(df)

        if 'best_reg' in filtered_df.columns:
            cond = (filtered_df.reg_C != augmented_df.best_reg) | \
                   (filtered_df.timepoint != augmented_df.best_timepoint)
            filtered_df.drop(filtered_df.loc[cond].index, inplace=True)

        else:
            names = list(df.name.unique())
            tasks = list(df.task.unique())

            for task in tasks:
                for name in names:
                    selected_df = augmented_df.loc[(augmented_df.name == name) & (augmented_df.task == task)]
                    if not len(selected_df):
                        continue

                    best_reg = selected_df.best_reg.unique()[0]
                    best_timepoint = selected_df.best_timepoint.unique()[0]

                    cond = (filtered_df.name == name) & \
                           (filtered_df.task == task) & \
                           ((filtered_df.reg_C != best_reg) | (filtered_df.timepoint != best_timepoint))
                    filtered_df.drop(filtered_df.loc[cond].index, inplace=True)

        filtered_dfs.append(reset_df(filtered_df))

    return filtered_dfs


def _detect_best_reg_timepoint(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=['best_reg', 'best_timepoint'], index=df.index)
    new_df = pd.concat([df, empty_df], axis=1)

    names = list(df.name.unique())
    tasks = list(df.task.unique())
    reg_cs = list(df.reg_C.unique())

    nb_c = len(reg_cs)
    nb_seeds = len(df.seed.unique())
    nt = len(df.timepoint.unique())

    for i, task in tqdm(enumerate(tasks), total=len(tasks),
                        desc='[PROGRESS] detecting best reg/timepoints', disable=not verbose):
        for j, name in enumerate(names):
            # select best reg
            cond = (df.name == name) & (df.task == task)
            selected_df = df.loc[cond]
            if not len(selected_df):
                continue

            scores = selected_df.loc[selected_df.metric.isin(['mcc', 'accuracy', 'f1']), 'score'].to_numpy()
            scores = scores.reshape(nb_seeds, nb_c, 3, nt)
            mean_scores = scores.mean(2).mean(0)

            confidences = selected_df.loc[selected_df.metric == 'confidence', 'score'].to_numpy()
            confidences = confidences.reshape(nb_seeds, nb_c, nt)
            mean_confidences = confidences.mean(0)

            max_score = np.max(mean_scores)

            if np.sum(mean_scores == max_score) == 1:
                a, b = np.unravel_index(np.argmax(mean_scores), mean_scores.shape)
            else:
                only_max_scores = mean_scores.copy()
                only_max_scores[mean_scores < max_score] = 0

                a = max(np.where(only_max_scores)[0])
                max_confidences = mean_confidences * (mean_scores == max_score)
                b = np.argmax(max_confidences[a])

            assert mean_scores[a, b] == np.max(mean_scores), "must select max score"

            new_df.loc[cond, 'best_reg'] = pd.Series([reg_cs[a]] * len(selected_df), index=selected_df.index)
            new_df.loc[cond, 'best_timepoint'] = pd.Series([b] * len(selected_df), index=selected_df.index)

    return reset_df(new_df)


def _normalize_confidence_score(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    names = list(df.name.unique())
    tasks = list(df.task.unique())
    reg_cs = list(df.reg_C.unique())

    for name in tqdm(names, desc='[PROGRESS] normalizing confidence scores', disable=not verbose):
        for task in tasks:
            for reg_c in reg_cs:
                cond = (df.name == name) & \
                       (df.task == task) & \
                       (df.reg_C == reg_c) & \
                       (df.metric == 'confidence')
                selected_df = df.loc[cond]

                if not len(selected_df):
                    continue

                confidence_scores = selected_df.score
                max_confidence = max(confidence_scores)
                df.loc[cond, 'score'] = confidence_scores / max_confidence

    return df


def _merg_dicts(dict_list: List[dict]) -> Dict[str, list]:
    merged = defaultdict(list)
    dict_items = map(methodcaller('items'), dict_list)
    for k, v in chain.from_iterable(dict_items):
        merged[k].extend(v)
    return merged


def _mk_save_dirs(cm: str, save_results_dir: str, classifier_args: Dict[str, str], verbose: bool = True):
    c_dir = ""
    for reg_c in classifier_args['C']:
        c_dir += "{},".format(reg_c)
    save_dir = pjoin(save_results_dir, classifier_args['clf_type'], cm, c_dir)
    os.makedirs(save_dir, exist_ok=True)

    coeffs_dir = pjoin(save_dir, '_coeffs')
    performances_dir = pjoin(save_dir, '_performances')
    classifiers_dir = pjoin(save_dir, '_classifiers')

    os.makedirs(coeffs_dir, exist_ok=True)
    os.makedirs(performances_dir, exist_ok=True)
    os.makedirs(classifiers_dir, exist_ok=True)

    if verbose:
        print("[PROGRESS] creaded save dirs")

    return save_dir, coeffs_dir, performances_dir, classifiers_dir


def _setup_logger(msg: str, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _dir = './log'
    os.makedirs(_dir, exist_ok=True)

    _file = "{:s}_{:s}.log".format(msg, now())
    logger_name = pjoin(_dir, _file)
    file_handler = logging.FileHandler(logger_name)
    if verbose:
        print("[PROGRESS] logger '{:s}' created".format(logger_name))

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def _setup_args() -> argparse.ArgumentParser:
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
        nargs='+',
        default=1.0,
    )
    parser.add_argument(
        "--clf_type",
        help="classifier type, choices: {'logreg', 'svm'}",
        type=str,
        choices={'logreg', 'svm'},
        default='logreg',
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
        default='10',
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
        choices={'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default='liblinear',
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
        default=500000,
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
        "--verbose", help="verbosity",
        action="store_true",
        )
    parser.add_argument(
        "--machine",
        help="machine name, choices = {'SigurRos', 'V1'}",
        type=str,
        choices={'SigurRos', 'V1'},
        default='SigurRos',
    )

    return parser.parse_args()


def main():
    args = _setup_args()

    if args.machine == 'SigurRos':
        base_dir = pjoin(os.environ['HOME'], 'Documents/PROJECTS/Kanold')
    elif args.machine == 'V1':
        base_dir = pjoin(os.environ['HOME'], 'Documents/Kanold')
    else:
        raise RuntimeError("invalid machine name: {}".format(args.machine))

    results_dir = pjoin(base_dir, 'results')
    processed_dir = pjoin(base_dir, 'python_processed')
    h_load_file = pjoin(processed_dir, "processed_data_nb_std={:d}.h5".format(args.nb_std))

    raw_tasks = ['hit', 'miss', 'correctreject', 'falsealarm']
    tasks = []
    for i, t1 in enumerate(raw_tasks):
        reduced = [item for item in raw_tasks[i:] if item != t1]
        for j, t2 in enumerate(reduced):
            tasks.append('{:s}/{:s}'.format(t1, t2))
    tasks += ['target/nontarget']
    seeds = [np.power(2, i) for i in range(args.nb_seeds)]

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # fit models
    fit_metadata = run_classification_analysis(
        load_file=h_load_file,
        cm=args.cm,
        save_results_dir=results_dir,
        tasks=tasks,
        seeds=seeds,
        xv_fold=args.xv_fold,
        save_to_pieces=args.save_to_pieces,
        verbose=args.verbose,
        clf_type=args.clf_type,
        penalty=args.penalty,
        C=args.C if isinstance(args.C, list) else [args.C],
        solver=args.solver,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    # combine fits together
    combine_fits(fit_metadata, verbose=args.verbose)


if __name__ == "__main__":
    main()
