import h5py
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime
from os.path import join as pjoin

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score, f1_score, make_scorer


def print_data_info():
    pass


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
        'type': 'logreg',
        'penalty': 'l1',
        'C': 0.05,
        'tol': 1e-4,
        'class_weight': 'balanced',
        'solver': 'liblinear',
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
        'name', 'seed', 'task', 'coeffs',
        'coeffs_normalized', 'coeffs_normalized_abs',
        'max_abs_val', 'percent_nonzero', 'best_score', 'best_timepoint',
    ]
    df_stats = pd.DataFrame(columns=cols)
    cols = [
        'name', 'seed', 'task', 'cell_indx', 'coeffs',
        'importances', 'x_coordinates', 'y_coordinates',
    ]
    df_data = pd.DataFrame(columns=cols)

    msg = "[INFO] running analysis using: '{}' criterion, {:d} different seeds.\n[INFO] classifier options: {}"
    msg = msg.format(criterion, len(seeds), classifier_args)
    print(msg)

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

            for task in tasks:
                try:
                    pos_lbl, neg_lbl = task.split('/')
                    pos = trial_info[pos_lbl]
                    neg = trial_info[neg_lbl]
                except KeyError:
                    if random_state == seeds[0]:
                        msg = 'missing trials in some trial type, name = {:s}, moving on . . .'
                        msg = msg.format(expt)
                        print(msg)
                    continue
                include_trials = np.logical_or(pos, neg)
                pos = pos[include_trials]
                neg = neg[include_trials]

                nb_pos_samples = sum(pos)
                nb_neg_samples = sum(neg)

                if nb_pos_samples == 0 or nb_neg_samples == 0:
                    if random_state == seeds[0]:
                        msg = 'no samples found for a trial type, name = {:s}, moving on . . .'
                        msg = msg.format(expt)
                        print(msg)
                    continue
                assert np.all(np.array([pos, neg]).T.sum(-1) == 1), "pos and neg labels should be mutually exclusive"

                # get xv indices
                xv_fold = 5
                pos_vld_indxs = random.sample(range(nb_pos_samples), int(np.ceil(nb_pos_samples / xv_fold)))
                neg_vld_indxs = random.sample(range(nb_neg_samples), int(np.ceil(nb_neg_samples / xv_fold)))

                pos_vld_indxs = np.where(pos)[0][pos_vld_indxs]
                neg_vld_indxs = np.where(neg)[0][neg_vld_indxs]

                vld_indxs = list(pos_vld_indxs) + list(neg_vld_indxs)
                trn_indxs = list(set(range(nb_pos_samples + nb_neg_samples)).difference(set(vld_indxs)))

                x = dff[:, include_trials, :]

                chosen_timepoint = 0
                chosen_performance = 0
                chosen_coeffs = np.zeros(nc)
                chosen_loglikelohoods = - np.inf * np.ones(len(vld_indxs))
                accepted = False
                for time_point in range(nt):

                    x_trn, x_vld = x[time_point][trn_indxs], x[time_point][vld_indxs]
                    y_trn, y_vld = pos[trn_indxs], pos[vld_indxs]
                    try:
                        if classifier_args['type'] == 'logreg':
                            clf = LogisticRegression(
                                penalty=classifier_args['penalty'],
                                C=classifier_args['C'],
                                tol=classifier_args['tol'],
                                solver=classifier_args['solver'],
                                class_weight=classifier_args['class_weight'],
                                n_jobs=classifier_args['n_jobs'],
                                random_state=random_state,
                            ).fit(x_trn, y_trn)
                        elif classifier_args['type'] == 'svm':
                            raise NotImplemented
                        else:
                            msg = "invalid classifier type encountered: {:s}, valid options are: {}"
                            msg = msg.format(classifier_args['type'], _allowed_clf_types)
                            raise ValueError(msg)
                        y_pred = clf.predict(x_vld)
                    except ValueError:
                        if random_state == seeds[0]:
                            msg = 'num trials too small, name = {:s}, task = {:s}, moving on . . .'
                            msg = msg.format(expt, task)
                            print(msg)
                        break

                    current_performance = score_fn(y_vld, y_pred)
                    current_loglikelihood = np.max(clf.predict_log_proba(x_vld), axis=-1)

                    accept_condition = (
                            current_performance > chosen_performance and
                            sum(current_loglikelihood) > sum(chosen_loglikelohoods) and
                            time_point > 30
                    )
                    if accept_condition:
                        chosen_performance = current_performance
                        chosen_loglikelohoods = current_loglikelihood
                        chosen_timepoint = time_point
                        chosen_coeffs = clf.coef_.copy().flatten()
                        accepted = True

                if accepted:
                    nonzero_coeffs = chosen_coeffs[chosen_coeffs != 0]
                    nb_nonzero = len(nonzero_coeffs)
                    max_abs_val = max(nonzero_coeffs, key=abs)
                    nonzero_coeffs_normalized = nonzero_coeffs / abs(max_abs_val)

                    data_dict = {
                        'name': [expt] * nb_nonzero,
                        'seed': [random_state] * nb_nonzero,
                        'task': [task] * nb_nonzero,
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

                    if classifier_args['type'] == 'logreg':
                        clf = LogisticRegression(
                            penalty=classifier_args['penalty'],
                            C=classifier_args['C'],
                            tol=classifier_args['tol'],
                            solver=classifier_args['solver'],
                            class_weight=classifier_args['class_weight'],
                            n_jobs=classifier_args['n_jobs'],
                            random_state=random_state,
                        ).fit(x_trn, y_trn)
                    elif classifier_args['type'] == 'svm':
                        raise NotImplemented
                    else:
                        msg = "invalid classifier type encountered: {:s}, valid options are: {}"
                        msg = msg.format(classifier_args['type'], _allowed_clf_types)
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
                        'cell_indx': range(nc),
                        'coeffs': chosen_coeffs,
                        'importances': importance_result['importances_mean'],
                        'x_coordinates': xy[:, 0],
                        'y_coordinates': xy[:, 1],
                    }
                    df_data = df_data.append(pd.DataFrame(data=data_dict))

                else:
                    print('No timepoint was accepted, name = {:s}, task = {:s}, moving on . . .'.format(expt, task))

    df_stats = df_stats.reset_index(drop=True)
    df_stats = df_stats.apply(pd.to_numeric, downcast="integer", errors="ignore")
    df_data = df_data.reset_index(drop=True)
    df_data = df_data.apply(pd.to_numeric, downcast="integer", errors="ignore")

    if save_results_dir is not None:
        save_file = "stats_{:s}.df".format(datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
        df_stats.to_pickle(pjoin(save_results_dir, save_file))
        save_file = "data_{:s}.df".format(datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
        df_data.to_pickle(pjoin(save_results_dir, save_file))

    h5_file.close()
    return df_stats, df_data


def _get_score_fn(option: str):
    return {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'f1': f1_score,
        'mcc': matthews_corrcoef,
    }[option]
