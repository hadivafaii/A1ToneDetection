import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin
from typing import List, Tuple, Union

from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, make_scorer

sys.path.append('..')
from utils.generic_utils import now, rm_dirs, merge_dicts, save_obj, smoothen


def combine_results(run_dir: str, regs_to_include: List[str] = None, verbose: bool = True):
    # sorts in increasing C value or: decreasing reg strength
    runs = sorted(os.listdir(run_dir), key=lambda c: float(c))
    if regs_to_include is not None:
        if not isinstance(regs_to_include, list):
            regs_to_include = [regs_to_include]
        runs = [item for item in runs if item in regs_to_include]
    if len(runs) == 0:
        raise RuntimeError("data not found")

    if verbose:
        print("[PROGRESS] using fits: {}".format(runs))

    coeffs_dictlist = []
    performances_dictlist = []
    for x in tqdm(runs, '[PROGRESS] combining previous fit data together', disable=not verbose):
        load_dir = pjoin(run_dir, x)
        files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
        listdir = os.listdir(load_dir)

        if not all(elem in listdir for elem in files):
            metadata = np.load(pjoin(load_dir, 'fit_metadata.npy'), allow_pickle=True).item()
            combine_fits(metadata, verbose)

        with open(pjoin(load_dir, '_coeffs.npy'), 'rb') as f:
            _coeffs = np.load(f.name, allow_pickle=True).item()
            coeffs_dictlist.append(_coeffs)
        with open(pjoin(load_dir, '_performances.npy'), 'rb') as f:
            _performances = np.load(f.name, allow_pickle=True).item()
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
        load_dir = pjoin(run_dir, x)
        files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
        listdir = os.listdir(load_dir)

        if not all(elem in listdir for elem in files):
            metadata = np.load(pjoin(load_dir, 'fit_metadata.npy'), allow_pickle=True).item()
            combine_fits(metadata, verbose)

        with open(pjoin(load_dir, '_classifiers.npy'), 'rb') as f:
            _classifiers = np.load(f.name, allow_pickle=True).item()
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


def combine_fits(fit_metadata: Union[int, float, complex], verbose: bool = True):
    files = ['_coeffs.npy', '_performances.npy', '_classifiers.npy']
    listdir = os.listdir(fit_metadata['save_dir'])

    if not all(elem in listdir for elem in files):
        # coeffs
        _coeffs_dictlist = []
        dirs = sorted(os.listdir(fit_metadata['coeffs_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _coeffs together', disable=not verbose):
            with open(pjoin(fit_metadata['coeffs_dir'], x), 'rb') as f:
                data_dict = np.load(f.name, allow_pickle=True).item()
                _coeffs_dictlist.append(data_dict)
        _coeffs = merge_dicts(_coeffs_dictlist, verbose)
        save_obj(_coeffs, "_coeffs.npy", fit_metadata['save_dir'], 'np', verbose)
        del _coeffs

        # performances
        _performances_dictlist = []
        dirs = sorted(os.listdir(fit_metadata['performances_dir']))
        for x in tqdm(dirs, '[PROGRESS] combining _performances together', disable=not verbose):
            with open(pjoin(fit_metadata['performances_dir'], x), 'rb') as f:
                data_dict = np.load(f.name, allow_pickle=True).item()
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
    if all(elem in listdir for elem in files):
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
    performances = _detect_best_reg_timepoint(performances, verbose=verbose)
    output += (performances,)
    filtered = _filter(performances, coeffs, verbose)
    output += tuple(filtered)
    return output


def _filter(performances: dict, coeffs: dict, verbose: bool = True) -> Tuple[dict, dict]:
    # first do performances
    cond = (performances['reg_C'] == performances['best_reg']) & \
           (performances['timepoint'] == performances['best_timepoint'])
    performances_filtered = {k: v[cond] for k, v in performances.items()}

    # do coeffs
    names = list(np.unique(performances['name']))
    tasks = list(np.unique(performances['task']))

    matching_indxs = []
    for task in tqdm(tasks, desc='[PROGRESS] filtering data', disable=not verbose, leave=False):
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


def _detect_best_reg_timepoint(
        performances: dict,
        criterion: str = 'mcc',
        start_time: int = 30,
        threshold: float = 0.9,
        filter_sz: int = 5,
        verbose: bool = True,) -> dict:

    criterion_options = {'mcc': 0, 'accuracy': 1, 'f1': 2}
    assert criterion in criterion_options
    metric_indx = criterion_options[criterion]

    names = np.unique(performances['name']).tolist()
    tasks = np.unique(performances['task']).tolist()
    reg_cs = np.unique(performances['reg_C']).tolist()

    nb_c = len(reg_cs)
    nb_seeds = len(np.unique(performances['seed']))
    nt = len(np.unique(performances['timepoint']))

    best_reg = np.array([-1.0] * len(performances['name']))
    best_timepoint = np.array([-1] * len(performances['name']))

    for name in tqdm(names, desc='[PROGRESS] detecting best reg/timepoints', disable=not verbose):
        for task in tqdm(tasks, disable=not verbose, leave=False):
            cond = (performances['name'] == name) & (performances['task'] == task)
            if not sum(cond):
                continue

            scores_all = np.array(performances['score'])[cond]
            scores_all = scores_all.reshape(nb_c, nb_seeds, 4, nt)

            scores = scores_all[..., metric_indx, :]
            confidences = scores_all[..., -1, :]

            mean_scores = scores.mean(1)
            mean_scores = smoothen(mean_scores, filter_sz=filter_sz)
            mean_confidences = confidences.mean(1)

            max_score = np.max(mean_scores[:, start_time:])
            above_threshold_bool = mean_scores > (threshold * max_score)
            above_threshold = mean_scores.copy()
            above_threshold[~above_threshold_bool] = 0

            a = min(np.unique(np.where(above_threshold_bool[:, start_time:])[0]), key=lambda x: reg_cs[x])
            max_confidences = mean_confidences * above_threshold_bool
            b = np.argmax(max_confidences[a][start_time:]) + start_time

            assert mean_scores[a, b] > (threshold * max_score), "must select max score"

            best_reg[cond] = reg_cs[a]
            best_timepoint[cond] = b

    assert not (best_reg < 0).sum(), "otherwise something wrong"
    assert not (best_timepoint < 0).sum(), "otherwise something wrong"

    performances['best_reg'] = best_reg
    performances['best_timepoint'] = best_timepoint

    return performances


def _compute_feature_importances(coeffs_filtered: dict, classifiers: dict, verbose: bool = True) -> dict:
    names = np.unique(coeffs_filtered['name']).tolist()
    tasks = np.unique(coeffs_filtered['task']).tolist()
    seeds = np.unique(coeffs_filtered['seed']).tolist()

    importances = np.array([-np.inf] * len(coeffs_filtered['name']))
    for name in tqdm(names, desc='[PROGRESS] computing feature importances', disable=not verbose):
        for task in tqdm(tasks, disable=not verbose, leave=False):
            cond = (coeffs_filtered['name'] == name) & (coeffs_filtered['task'] == task)

            if not sum(cond):
                continue
            _c = np.unique(coeffs_filtered['reg_C'][cond])
            _t = np.unique(coeffs_filtered['timepoint'][cond])
            assert len(_c) == len(_t) == 1,\
                "filtered df must have only one unique selected timepoint or reg per expt/task"
            best_reg = _c.item()
            best_timepoint = _t.item()

            _importances = []
            for random_state in sorted(seeds):
                k = "{}^{}^{}^{}^{}".format(name, task, random_state, best_reg, best_timepoint)
                clf, x_vld, y_vld = classifiers[k]
                importance_result = permutation_importance(
                    estimator=clf,
                    X=x_vld,
                    y=y_vld,
                    n_repeats=100,
                    n_jobs=-1,
                    scoring=make_scorer(matthews_corrcoef),
                    random_state=random_state,
                )
                _importances.extend(importance_result.importances_mean)
            importances[cond] = _importances

    assert not np.isinf(importances).sum(), "otherwise something wrong"
    coeffs_filtered['importances'] = importances
    return coeffs_filtered


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cm",
        help="a comment about this fit",
        type=str,
    )
    parser.add_argument(
        "--clf_type",
        help="classifier type, choices: {'logreg', 'svm'}",
        type=str,
        choices={'logreg', 'svm'},
        default='svm',
    )
    parser.add_argument(
        '--regs_to_include',
        help='list of regularizations to include in the combined result',
        type=float,
        nargs='+',
        default=None,
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
    run_dir = pjoin(results_dir, args.clf_type, args.cm)

    # combine fits together
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        combine_results(run_dir, args.regs_to_include, verbose=args.verbose)

    print("[PROGRESS] done.\n")


if __name__ == "__main__":
    main()
