import os
import h5py
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
from os.path import join as pjoin

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score, f1_score, make_scorer


# TODO: add the actual time analysis here too (it's just adding another dataframe)
def run_classification_analysis(
        load_file: str,
        tasks: List[str] = None,
        seeds: List[int] = 42,
        criterion: str = 'mcc',
        save_results_dir: str = None,
        **kwargs: Dict[str, str],
):

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
        'n_jobs': None,
    }
    for k in classifier_args:
        if k in kwargs:
            classifier_args[k] = kwargs[k]

    _allowed_clf_types = ['logreg', 'svm']
    score_fn = _get_score_fn(criterion)

    if not isinstance(seeds, list):
        seeds = [seeds]

    cols = [
        'name', 'seed', 'task', 'reg_C',
        'coeffs', 'coeffs_normalized', 'coeffs_normalized_abs',
        'max_abs_val', 'percent_nonzero', 'best_score', 'best_timepoint',
    ]
    df_stats = pd.DataFrame(columns=cols)
    cols = [
        'name', 'seed', 'task', 'reg_C',
        'cell_indx', 'coeffs', 'importances',
        'x_coordinates', 'y_coordinates',
    ]
    df_results_concise = pd.DataFrame(columns=cols)
    cols = [
        'name', 'seed', 'task', 'reg_C',
        'timepoint', 'metric', 'score',
    ]
    df_results_extensive = pd.DataFrame(columns=cols)

    xv_fold = 5
    msg = "[INFO] running analysis using: '{}' criterion, {:d}-fold xv, {:d} different seeds.\n\
    [INFO] classifier options: {}"
    msg = msg.format(criterion, xv_fold, len(seeds), classifier_args)
    print(msg)

    logger = _setup_logger(classifier_args['clf_type'])

    h5_file = h5py.File(load_file, "r")
    pbar = tqdm(h5_file)
    for expt in pbar:
        behavior = h5_file[expt]["behavior"]
        trial_info_grp = behavior["trial_info"]

        good_cells = np.array(behavior["good_cells"], dtype=int)
        xy = np.array(behavior["xy"], dtype=float)[good_cells]
        dff = np.array(behavior["dff"], dtype=float)[..., good_cells]
        nt, nb_trials, nc = dff.shape

        trial_info = {}
        for k, v in trial_info_grp.items():
            trial_info[k] = np.array(v, dtype=int)

        for random_state in seeds:
            random.seed(random_state)
            np.random.seed(random_state)

            for reg_c in classifier_args['C']:
                for task in tasks:
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

                    chosen_timepoint = 0
                    chosen_performance = - np.inf
                    chosen_coeffs = np.zeros(nc)
                    chosen_confidence = 0
                    accepted = False

                    mcc_all = np.zeros(nt)
                    accuracy_all = np.zeros(nt)
                    f1_all = np.zeros(nt)

                    for time_point in range(nt):
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
                                    n_jobs=classifier_args['n_jobs'],
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
                        except ValueError:
                            msg = 'num trials too small, name = {:s}, seed = {:d}, C = {}, task = {}'
                            msg = msg.format(expt, random_state, reg_c, task)
                            logger.info(msg)
                            break

                        mcc_all[time_point] = matthews_corrcoef(y_vld, y_pred)
                        accuracy_all[time_point] = accuracy_score(y_vld, y_pred)
                        f1_all[time_point] = f1_score(y_vld, y_pred)

                        current_performance = score_fn(y_vld, y_pred)
                        _confidence = clf.decision_function(x_vld)
                        current_confidence = sum(abs(_confidence[y_vld == y_pred]))

                        _cond = False
                        if current_performance > chosen_performance:
                            _cond = True
                        elif current_performance == chosen_performance:
                            if current_confidence > chosen_confidence:
                                _cond = True

                        if _cond:
                            chosen_performance = current_performance
                            chosen_confidence = current_confidence
                            chosen_timepoint = time_point
                            chosen_coeffs = clf.coef_.copy().flatten()
                            accepted = True

                    data_dict = {
                        'name': [expt] * 3 * nt,
                        'seed': [random_state] * 3 * nt,
                        'task': [task] * 3 * nt,
                        'reg_C': [reg_c] * 3 * nt,
                        'timepoint': np.tile(range(nt), 3),
                        'metric': ['mcc'] * nt + ['accuracy'] * nt + ['f1'] * nt,
                        'score': np.concatenate([mcc_all, accuracy_all, f1_all]),
                    }
                    df_results_extensive = df_results_extensive.append(pd.DataFrame(data=data_dict))

                    if accepted:
                        nonzero_coeffs = chosen_coeffs[chosen_coeffs != 0]
                        nb_nonzero = len(nonzero_coeffs)
                        if nb_nonzero == 0:
                            msg = 'all coeffs zero, name = {:s}, seed = {:d}, C = {}, task = {}'
                            msg = msg.format(expt, random_state, reg_c, task)
                            logger.warning(msg)
                            continue
                        max_abs_val = max(nonzero_coeffs, key=abs)
                        nonzero_coeffs_normalized = nonzero_coeffs / abs(max_abs_val)

                        data_dict = {
                            'name': [expt] * nb_nonzero,
                            'seed': [random_state] * nb_nonzero,
                            'task': [task] * nb_nonzero,
                            'reg_C': [reg_c] * nb_nonzero,
                            'coeffs': nonzero_coeffs,
                            'coeffs_normalized': nonzero_coeffs_normalized,
                            'coeffs_normalized_abs': abs(nonzero_coeffs_normalized),
                            'max_abs_val': [max_abs_val] * nb_nonzero,
                            'percent_nonzero': [np.round(nb_nonzero / nc * 100, decimals=1)] * nb_nonzero,
                            'best_score': [chosen_performance] * nb_nonzero,
                            'best_timepoint': [chosen_timepoint] * nb_nonzero,
                        }
                        df_stats = df_stats.append(pd.DataFrame(data=data_dict))

                        # get importance results
                        x_trn, x_vld = x[chosen_timepoint][trn_indxs], x[chosen_timepoint][vld_indxs]
                        y_trn, y_vld = pos[trn_indxs], pos[vld_indxs]

                        if classifier_args['clf_type'] == 'logreg':
                            clf = LogisticRegression(
                                penalty=classifier_args['penalty'],
                                C=reg_c,
                                tol=classifier_args['tol'],
                                solver=classifier_args['solver'],
                                class_weight=classifier_args['class_weight'],
                                n_jobs=classifier_args['n_jobs'],
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

                        importance_result = permutation_importance(
                            estimator=clf,
                            X=x_vld,
                            y=y_vld,
                            n_repeats=100,
                            n_jobs=-1,
                            scoring=make_scorer(score_fn),
                            random_state=random_state,
                        )

                        data_dict = {
                            'name': [expt] * nc,
                            'seed': [random_state] * nc,
                            'task': [task] * nc,
                            'reg_C': [reg_c] * nc,
                            'cell_indx': range(nc),
                            'coeffs': chosen_coeffs,
                            'importances': importance_result['importances_mean'],
                            'x_coordinates': xy[:, 0],
                            'y_coordinates': xy[:, 1],
                        }
                        df_results_concise = df_results_concise.append(pd.DataFrame(data=data_dict))

                    else:
                        msg = 'no time points were accepted, name = {:s}, seed = {:d}, C = {}, task = {}'
                        msg = msg.format(expt, random_state, reg_c, task)
                        logger.warning(msg)

    df_stats = df_stats.reset_index(drop=True)
    df_stats = df_stats.apply(pd.to_numeric, downcast="integer", errors="ignore")
    df_results_concise = df_results_concise.reset_index(drop=True)
    df_results_concise = df_results_concise.apply(pd.to_numeric, downcast="integer", errors="ignore")
    df_results_extensive = df_results_extensive.reset_index(drop=True)
    df_results_extensive = df_results_extensive.apply(pd.to_numeric, downcast="integer", errors="ignore")

    if save_results_dir is not None:
        save_dir = pjoin(save_results_dir, classifier_args['clf_type'])
        os.makedirs(save_dir, exist_ok=True)

        name_template = "{:s}{:s}_{:s}.df"
        name_template = name_template.format(
            classifier_args['penalty'],
            classifier_args['solver'],
            datetime.now().strftime("[%Y_%m_%d_%H:%M]"),
        )

        file_name = "stats_" + name_template
        df_stats.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "concise_" + name_template
        df_results_concise.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

        file_name = "extensive_" + name_template
        df_results_extensive.to_pickle(pjoin(save_dir, file_name))
        print("[PROGRESS] '{:s}' saved at {:s}".format(file_name, save_dir))

    h5_file.close()
    return df_stats, df_results_concise, df_results_extensive


def _get_score_fn(option: str):
    return {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef,
    }[option]


def _setup_logger(msg):
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger_name = "./{:s}_{:s}.log".format(msg, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
    file_handler = logging.FileHandler(logger_name)
    print("[INFO] logger '{:s}' created".format(logger_name))

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
