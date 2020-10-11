import os
import sys
import h5py
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from os.path import join as pjoin
from typing import List, Tuple, Dict

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, make_scorer

sys.path.append('..')
from utils.generic_utils import now, reset_df, merge_dicts, rm_dirs, save_obj


def run_classification_analysis(
        cm: str,
        load_file: str,
        results_dir: str,
        tasks: List[str],
        seeds: List[int] = 42,
        xv_fold: int = 5,
        save_to_pieces: bool = False,
        verbose: bool = True,
        **kwargs: Dict[str, str], ) -> Tuple[pd.DataFrame]:
    _allowed_clf_types = ['logreg', 'svm']

    if not isinstance(seeds, list):
        seeds = [seeds]

    classifier_args = {
        'clf_type': 'logreg',
        'penalty': 'l1',
        'C': 1.0,
        'tol': 1e-4,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'max_iter': int(1e6),
    }
    for k in classifier_args:
        if k in kwargs:
            classifier_args[k] = kwargs[k]

    if verbose:
        msg = "\n[INFO] running analysis using: {:d}-fold xv, {:d} different seeds.\n"
        msg += "[INFO] classifier options:\n\t{}\n"
        msg += "[INFO] option save_to_pieces is: {}"
        msg = msg.format(xv_fold, len(seeds), classifier_args, save_to_pieces)
        print(msg)

    logger = _setup_logger(classifier_args['clf_type'])
    save_dir, coeffs_dir, performances_dir, classifiers_dir = _mk_save_dirs(
        cm, results_dir, classifier_args, verbose)

    coeffs_dict_list = []
    performances_dict_list = []
    _classifiers = {}
    counter = 0

    h5_file = h5py.File(load_file, "r")
    pbar = tqdm(h5_file, disable=not verbose, dynamic_ncols=True)
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

            for task in tqdm(tasks, leave=False, disable=not verbose):
                try:
                    pos_lbl, neg_lbl = task.split('/')
                    pos = trial_info[pos_lbl]
                    neg = trial_info[neg_lbl]
                except KeyError:
                    if random_state == seeds[0]:
                        msg = 'missing trials, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, classifier_args['C'], task)
                        logger.info(msg)
                    continue

                include_trials = np.logical_or(pos, neg)
                pos = pos[include_trials]
                neg = neg[include_trials]

                nb_pos_samples = sum(pos)
                nb_neg_samples = sum(neg)

                if nb_pos_samples == 0 or nb_neg_samples == 0:
                    if random_state == seeds[0]:
                        msg = 'no samples found, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, classifier_args['C'], task)
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
                                C=classifier_args['C'],
                                tol=classifier_args['tol'],
                                solver=classifier_args['solver'],
                                class_weight=classifier_args['class_weight'],
                                max_iter=classifier_args['max_iter'],
                                random_state=random_state,
                            ).fit(x_trn, y_trn)
                        elif classifier_args['clf_type'] == 'svm':
                            clf = LinearSVC(
                                penalty=classifier_args['penalty'],
                                C=classifier_args['C'],
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
                        k = "{}^{}^{}^{}^{}".format(expt, task, random_state, classifier_args['C'], time_point)
                        if save_to_pieces:
                            data_dict = {k: (clf, x_vld, y_vld)}
                            save_obj(data_dict, '{:09d}.npy'.format(counter), classifiers_dir, 'np', verbose=False)
                        else:
                            _classifiers[k] = (clf, x_vld, y_vld)

                    except ValueError:
                        msg = 'num trials too small, name = {:s}, seed = {:d}, C = {}, task = {}, t = {}'
                        msg = msg.format(expt, random_state, classifier_args['C'], task, time_point)
                        logger.info(msg)
                        break

                    mcc_all[time_point] = matthews_corrcoef(y_vld, y_pred)
                    accuracy_all[time_point] = accuracy_score(y_vld, y_pred)
                    f1_all[time_point] = f1_score(y_vld, y_pred)

                    confidence = clf.decision_function(x_vld)
                    confidence_all[time_point] = sum(abs(confidence[y_vld == y_pred]))

                    coeffs = clf.coef_.squeeze()
                    nb_nonzero = sum(coeffs != 0.0)

                    data_dict = {
                        'name': [expt] * nc,
                        'seed': [random_state] * nc,
                        'task': [task] * nc,
                        'reg_C': [classifier_args['C']] * nc,
                        'timepoint': [time_point] * nc,
                        'cell_indx': range(nc),
                        'coeffs': coeffs,
                        'nb_nonzero': [nb_nonzero] * nc,
                        'percent_nonzero': [nb_nonzero / nc * 100] * nc,
                        'x': xy[:, 0],
                        'y': xy[:, 1],
                    }
                    if save_to_pieces:
                        save_obj(data_dict, '{:09d}.npy'.format(counter), coeffs_dir, 'np', verbose=False)
                    else:
                        coeffs_dict_list.append(data_dict)

                    msg = "name: {}, seed: {}, task: {}, t: {}, "
                    msg = msg.format(expt, random_state, task, time_point)
                    pbar.set_description(msg)

                confidence_all /= np.maximum(1e-8, max(confidence_all))
                data_dict = {
                    'name': [expt] * nt * 4,
                    'seed': [random_state] * nt * 4,
                    'task': [task] * nt * 4,
                    'reg_C': [classifier_args['C']] * nt * 4,
                    'timepoint': np.tile(range(nt), 4),
                    'metric': ['mcc'] * nt + ['accuracy'] * nt + ['f1'] * nt + ['confidence'] * nt,
                    'score': np.concatenate([mcc_all, accuracy_all, f1_all, confidence_all]),
                }
                nan_detected = any(map(
                    lambda z: False if all(isinstance(item, str) for item in z) else any(np.isnan(z)),
                    data_dict.values()
                ))
                if not nan_detected:
                    if save_to_pieces:
                        save_obj(data_dict, '{:09d}.npy'.format(counter), performances_dir, 'np', verbose=False)
                    else:
                        performances_dict_list.append(data_dict)
                else:
                    msg = 'nan detected in performances data_dict, name = {:s}, seed = {:d}, C = {}, task = {}'
                    msg = msg.format(expt, random_state, classifier_args['C'], task)
                    logger.warning(msg)
    h5_file.close()

    fit_metadata = {
        'classifier_args': classifier_args,
        'save_dir': save_dir,
        'coeffs_dir': coeffs_dir,
        'performances_dir': performances_dir,
        'classifiers_dir': classifiers_dir,
        'datetime': now(),
    }
    save_obj(fit_metadata, 'fit_metadata.npy', save_dir, 'np', verbose)

    if not save_to_pieces:
        _coeffs = merge_dicts(coeffs_dict_list, verbose)
        _performances = merge_dicts(performances_dict_list, verbose)

        nan_detected = any(map(
            lambda z: False if all(isinstance(item, str) for item in z) else any(np.isnan(z)),
            _coeffs.values()
        ))
        if nan_detected:
            msg = 'nan detected in _coeffs'
            logger.warning(msg)

        nan_detected = any(map(
            lambda z: False if all(isinstance(item, str) for item in z) else any(np.isnan(z)),
            _performances.values()
        ))
        if nan_detected:
            msg = 'nan detected in _performances'
            logger.warning(msg)

        # save
        save_obj(_coeffs, "_coeffs.npy", fit_metadata['save_dir'], 'np', verbose)
        save_obj(_performances, "_coeffs.npy", fit_metadata['save_dir'], 'np', verbose)
        save_obj(_classifiers, "_coeffs.npy", fit_metadata['save_dir'], 'np', verbose)

    return fit_metadata


def combine_results(run_dir: str, verbose: bool = True) -> Tuple[pd.DataFrame]:
    _regs = ['1.0', '0.001', '0.0001']

    coeffs_dictlist = []
    performances_dictlist = []
    runs = sorted(os.listdir(run_dir), key=lambda c: float(c), reverse=True)
    for x in tqdm(runs, '[PROGRESS] combining previous fit data together', disable=not verbose):
        if x not in _regs:
            continue
        load_dir = pjoin(run_dir, x)
        files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
        listdir = os.listdir(load_dir)
        cond = set(files).issubset(set(listdir))

        if not cond:
            metadata = np.load(pjoin(load_dir, 'fit_metadata.npy'), allow_pickle=True).item()
            _combine_fits(metadata, verbose)

        with open(pjoin(load_dir, '_coeffs.npy'), 'rb') as f:
            _coeffs = np.load(f, allow_pickle=True).item()
            coeffs_dictlist.append(_coeffs)
        with open(pjoin(load_dir, '_performances.npy'), 'rb') as f:
            _performances = np.load(f, allow_pickle=True).item()
            performances_dictlist.append(_performances)

    coeffs = merge_dicts(coeffs_dictlist, verbose)
    performances = merge_dicts(performances_dictlist, verbose)

    performances, performances_filtered, coeffs_filtered = _porocess_results(
        performances, coeffs, verbose)

    # save
    time_now = now(exclude_hour_min=True)

    save_obj(
        data=pd.DataFrame.from_dict(coeffs),
        file_name="coeffs_{:s}.df".format(time_now),
        save_dir=run_dir,
        mode='df',
        verbose=verbose,
    )
    del coeffs
    save_obj(
        data=pd.DataFrame.from_dict(performances),
        file_name="performances_{:s}.df".format(time_now),
        save_dir=run_dir,
        mode='df',
        verbose=verbose,
    )
    del performances
    save_obj(
        data=pd.DataFrame.from_dict(performances_filtered),
        file_name="performances_filtered_{:s}.df".format(time_now),
        save_dir=run_dir,
        mode='df',
        verbose=verbose,
    )
    del performances_filtered

    # clfs
    classifiers = {}
    for x in tqdm(runs, '[PROGRESS] combining previous classifiers together', disable=not verbose):
        if x not in _regs2:
            continue
        load_dir = pjoin(run_dir, x)
        files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
        listdir = os.listdir(load_dir)
        cond = set(files).issubset(set(listdir))

        if not cond:
            metadata = np.load(pjoin(load_dir, 'fit_metadata.npy'), allow_pickle=True).item()
            _combine_fits(metadata, verbose)

        with open(pjoin(load_dir, '_classifiers.npy'), 'rb') as f:
            _classifiers = np.load(f, allow_pickle=True).item()
            assert not set(_classifiers.keys()).intersection(set(classifiers.keys())),\
                "must have non-overlapping keys by design"
            classifiers.update(_classifiers)

    coeffs_filtered = _compute_feature_importances(coeffs_filtered, classifiers)

    # save
    save_obj(
        data=pd.DataFrame.from_dict(coeffs_filtered),
        file_name="coeffs_filtered_{:s}.df".format(time_now),
        save_dir=run_dir,
        mode='df',
        verbose=verbose,
    )
    del coeffs_filtered
    save_obj(classifiers, "classifiers_{:s}.pkl".format(time_now), run_dir, 'pkl', verbose)
    del classifiers

    print("[PROGRESS] combining everything done.\n\n")


def _combine_fits(fit_metadata: Dict[str, str], verbose: bool = True):
    files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
    listdir = os.listdir(fit_metadata['save_dir'])
    cond = set(files).issubset(set(listdir))

    if not cond:
        # coeffs
        _coeffs_dictlist = []
        dirs = sorted(os.listdir(fit_metadata['coeffs_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _coeffs together', disable=not verbose):
            with open(pjoin(fit_metadata['coeffs_dir'], x), 'rb') as f:
                data_dict = np.load(f, allow_pickle=True).item()
                _coeffs_dictlist.append(data_dict)
        _coeffs = merge_dicts(_coeffs_dictlist, verbose)
        save_obj(_coeffs, "_coeffs.npy", fit_metadata['save_dir'], 'np', verbose)
        del _coeffs

        # performances
        _performances_dictlist = []
        dirs = sorted(os.listdir(fit_metadata['performances_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _performances together', disable=not verbose):
            with open(pjoin(fit_metadata['performances_dir'], x), 'rb') as f:
                data_dict = np.load(f, allow_pickle=True).item()
                _performances_dictlist.append(data_dict)
        _performances = merge_dicts(_performances_dictlist, verbose)
        save_obj(_performances, "_performances.npy", fit_metadata['save_dir'], 'np', verbose)
        del _performances

        # classifiers
        _classifiers = {}
        dirs = sorted(os.listdir(fit_metadata['classifiers_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _classifiers together', disable=not verbose):
            with open(pjoin(fit_metadata['classifiers_dir'], x), 'rb') as f:
                data_dict = np.load(f, allow_pickle=True).item()
                _classifiers.update(data_dict)
        save_obj(_classifiers, "_classifiers.npy", fit_metadata['save_dir'], 'np', verbose)
        del _classifiers

    else:
        if verbose:
            print('[PROGRESS] skipped combining, files found: {}'.format(files))

    # delete files
    listdir = os.listdir(fit_metadata['save_dir'])
    cond = set(files).issubset(set(listdir))
    if cond:
        dirs = [
            fit_metadata['coeffs_dir'].split('/')[-1],
            fit_metadata['performances_dir'].split('/')[-1],
            fit_metadata['classifiers_dir'].split('/')[-1],
        ]
        rm_dirs(fit_metadata['save_dir'], dirs, verbose)
    else:
        print("[WARNING] some fits were not combined here: {}".format(fit_metadata['save_dir']))


def _porocess_results(performances: dict, coeffs: dict, verbose: bool = True) -> tuple:
    performances = {k: np.array(v) for k, v in performances.items()}
    coeffs = {k: np.array(v) for k, v in coeffs.items()}

    output = ()
    performances = _detect_best_reg_timepoint(performances, verbose)
    output += (performances,)
    filtered = _filter(performances, coeffs, verbose)
    output += tuple(filtered)
    return output


def _filter(performances: dict, coeffs: dict, verbose: bool = True) -> Tuple[dict]:
    # first do performances
    cond = (performances['reg_C'] == performances['best_reg']) & \
           (performances['timepoint'] == performances['best_timepoint'])
    performances_filtered = {k: v[cond] for k, v in performances.items()}

    # do coeffs
    names = list(np.unique(performances['name']))
    tasks = list(np.unique(performances['task']))

    matching_indxs = []
    for task in tqdm(tasks, disable=not verbose, leave=False):
        for name in names:
            cond = (performances['name'] == name) & (performances['task'] == task)
            if not sum(cond):
                continue

            best_reg = np.unique(performances['best_reg'][cond]).item()
            best_timepoint = np.unique(performances['best_timepoint'][cond]).item()

            cond = (coeffs['name'] == name) & (coeffs['task'] == task) & \
                   (coeffs['reg_C'] == best_reg) & (coeffs['timepoint'] == best_timepoint)
            matching_indxs.extend(cond.nonzero()[0])

    coeffs_filtered = {k: v[matching_indxs] for k, v in coeffs.items()}
    return performances_filtered, coeffs_filtered


def _detect_best_reg_timepoint(performances: dict, verbose: bool = True) -> dict:
    names = list(np.unique(performances['name']))
    tasks = list(np.unique(performances['task']))
    reg_cs = list(np.unique(performances['reg_C']))

    nb_c = len(reg_cs)
    nb_seeds = len(np.unique(performances['seed']))
    nt = len(np.unique(performances['timepoint']))

    best_reg = np.array([-1.0] * len(performances['name']))
    best_timepoint = np.array([-1] * len(performances['name']))
    for i, task in tqdm(enumerate(tasks), total=len(tasks),
                        desc='[PROGRESS] detecting best reg/timepoints', disable=not verbose):
        for j, name in enumerate(names):
            # select best reg
            cond = (performances['name'] == name) & (performances['task'] == task)
            if not sum(cond):
                continue

            scores_all = np.array(performances['score'])[cond]
            scores_all = scores_all.reshape(nb_seeds, nb_c, 4, nt)

            scores = scores_all[..., :3, :]
            confidences = scores_all[..., -1, :]

            mean_scores = scores.mean(2).mean(0)
            mean_confidences = confidences.mean(0)

            max_score = np.max(mean_scores)

            if np.sum(mean_scores == max_score) == 1:
                a, b = np.unravel_index(np.argmax(mean_scores), mean_scores.shape)
            else:
                only_max_scores = mean_scores.copy()
                only_max_scores[mean_scores < max_score] = 0

                # a will correspond to smaller C -> larger reg strength
                a = min(np.unique(np.where(only_max_scores)[0]), key=lambda x: reg_cs[x])
                max_confidences = mean_confidences * (mean_scores == max_score)
                b = np.argmax(max_confidences[a])

            assert mean_scores[a, b] == np.max(mean_scores), "must select max score"

            best_reg[cond] = reg_cs[a]
            best_timepoint[cond] = b

    assert not (best_reg < 0).sum(), "otherwise something wrong"
    assert not (best_timepoint < 0).sum(), "otherwise something wrong"

    performances['best_reg'] = best_reg
    performances['best_timepoint'] = best_timepoint

    return performances


def _compute_feature_importances(coeffs_filtered: dict, classifiers: dict, verbose: bool = True) -> dict:
    importances = np.array([-1.0] * len(coeffs_filtered))
    for k, (clf, x_vld, y_vld) in tqdm(
            classifiers.items(), desc='[PROGRESS] computing feature importances', disable=not verbose):
        name, task, random_state, reg_c, time_point = k.split('^')
        random_state, time_point = int(random_state), int(time_point)
        reg_c = float(reg_c)

        cond = (coeffs_filtered['name'] == name) & \
               (coeffs_filtered['task'] == task) & \
               (coeffs_filtered['seed'] == random_state)

        _c = np.unique(coeffs_filtered['reg_C'][cond])
        _t = np.unique(coeffs_filtered['timepoint'][cond])
        assert len(_c) == len(_t) == 1, "filtered df must have only one unique selected timepoint or reg per expt/task"

        if not (_c.item() == reg_c and _t.item() == time_point):
            continue

        importance_result = permutation_importance(
            estimator=clf,
            X=x_vld,
            y=y_vld,
            n_repeats=10,
            n_jobs=-1,  # TODO: check this tomorrow morning
            scoring=make_scorer(matthews_corrcoef),
            random_state=random_state,
        )
        importances[cond] = importance_result['importances_mean']

    assert not (importances < 0).sum(), "otherwise something wrong"
    coeffs_filtered['importances'] = importances

    return coeffs_filtered


def _mk_save_dirs(cm: str, results_dir: str, classifier_args: Dict[str, str], verbose: bool = True):
    c_dir = "{}".format(classifier_args['C'])
    save_dir = pjoin(results_dir, classifier_args['clf_type'], cm, c_dir)
    os.makedirs(save_dir, exist_ok=True)

    coeffs_dir = pjoin(save_dir, '_coeffs')
    performances_dir = pjoin(save_dir, '_performances')
    classifiers_dir = pjoin(save_dir, '_classifiers')

    os.makedirs(coeffs_dir, exist_ok=True)
    os.makedirs(performances_dir, exist_ok=True)
    os.makedirs(classifiers_dir, exist_ok=True)

    if verbose:
        print("[PROGRESS] creaded save dirs\n")

    return save_dir, coeffs_dir, performances_dir, classifiers_dir


def _setup_logger(msg: str, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    _dir = './log'
    os.makedirs(_dir, exist_ok=True)

    _file = "{:s}_{:s}.log".format(msg, now(exclude_hour_min=True))
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
    h_load_file = pjoin(processed_dir, "organized_nb_std={:d}.h5".format(args.nb_std))

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

    seeds = [np.power(2, i) for i in range(args.nb_seeds)]

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # fit models
    fit_metadata = run_classification_analysis(
        cm=args.cm,
        load_file=h_load_file,
        results_dir=results_dir,
        tasks=tasks,
        seeds=seeds,
        xv_fold=args.xv_fold,
        save_to_pieces=args.save_to_pieces,
        verbose=args.verbose,
        clf_type=args.clf_type,
        penalty=args.penalty,
        C=args.C,
        solver=args.solver,
        tol=args.tol,
        max_iter=args.max_iter,
    )

    # combine fits together
    _combine_fits(fit_metadata, verbose=args.verbose)
    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
