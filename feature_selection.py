import contextlib
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn import linear_model, pipeline, preprocessing, metrics

from helper import evaluate_performance

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


def meanstd_str(arr, prec=2) -> str:
    return f"{np.nanmean(arr):.{prec}f}±{np.nanstd(arr):.{prec}f}"


def get_pipeline():
    return pipeline.Pipeline(
        [
            ("scale", preprocessing.StandardScaler()),
            (
                "predict",
                linear_model.LogisticRegression(C=1e-3, class_weight="balanced"),
            ),
            ### Robustness Checks
            # ('predict', linear_model.LogisticRegression(C=1e-3, class_weight='balanced')), # Ridge
            # ('predict', linear_model.LogisticRegression(C=1e-3, penalty='l1', solver='saga', class_weight='balanced')), # Lasso
            # ('predict', linear_model.LogisticRegression(C=1e-3, l1_ratio=0.5, penalty='elasticnet', solver='saga', class_weight='balanced')), # Elasticnet
            # ('predict', neural_network.MLPClassifier(activation='logistic', max_iter=50)), # MLP
            # ('predict', ensemble.GradientBoostingClassifier(n_estimators=10, min_samples_split=10, max_depth=3)), # Gradient boosting
            # ('predict', linear_model.LogisticRegression(penalty='none', class_weight='balanced')),
            # ('predict', linear_model.LogisticRegression(C=1e0, class_weight='balanced')),
            # ('predict', linear_model.LogisticRegression(C=1e0, penalty='l1', solver='saga', class_weight='balanced')),
            # ('predict', linear_model.LogisticRegression(C=1e-3, l1_ratio=0.5, penalty='elasticnet', solver='saga', class_weight='balanced')), # Elasticnet
            # ('predict', lgb.LGBMClassifier(verbose=-1, is_unbalance=True, boost_from_average=False, num_leaves=5, num_trees=100, min_data_in_leaf=100))
            # ('predict', lgb.LGBMClassifier(n_estimators=20, min_child_samples=10, num_leaves=5, max_depth=3, objective='binary', boost_from_average=True, n_jobs=4))
            # ('predict', lgb.LGBMClassifier(verbose=-1,  boosting_type='rf', subsample_freq=2, subsample=0.2, min_child_samples=100, max_depth=10, n_estimators=10, num_leaves=5, objective='binary', boost_from_average=False, n_jobs=4))
        ]
    )


def calculate_cv_score(
    X,
    y,
    features,
    groups,
    scenarios,
    train_indices,
    test_indices,
    desc,
    bg=None,
    store_files=False,
    pipe=None,
):
    train_type = (
        "moderate"
        if np.all(np.unique(groups[train_indices]) < 300)
        else "mild"
        if np.all(np.unique(groups[train_indices]) > 300)
        else "mixed"
    )
    eval_type = (
        "moderate"
        if np.all(np.unique(groups[test_indices]) < 300)
        else "mild"
        if np.all(np.unique(groups[test_indices]) > 300)
        else "mixed"
    )
    print(
        f"### Running {desc.upper()}: train {train_type.upper()}, evaluate {eval_type.upper()}."
    )
    # print(f'Train subjects (n = {len(groups[train_indices].unique())}): {sorted(groups[train_indices].unique())}')
    # print(f'Test subjects (n = {len(groups[test_indices].unique())}): {sorted(groups[test_indices].unique())}')
    print(f"Features (n = {len(features)}): {features}")

    pipe = get_pipeline() if pipe is None else pipe
    aucs = []
    results = pd.DataFrame(
        {
            "id": groups[test_indices],
            "y_test": y[test_indices],
            "scenario": scenarios[test_indices],
        },
        index=y[test_indices].index,
    )
    coefs = pd.DataFrame(index=pd.Index(np.unique(groups[test_indices]), name="id"))

    # fit only once if no test subject in train sets
    fitted = False
    if (
        len(
            np.intersect1d(
                np.unique(groups[train_indices]), np.unique(groups[test_indices])
            )
        )
        == 0
    ):
        pipe.fit(X[train_indices][features], y[train_indices])
        fitted = True

    subjects = (
        np.unique(groups[test_indices])
        if fitted
        else tqdm(np.unique(groups[test_indices]))
    )

    for group in subjects:
        if not fitted:
            pipe.fit(
                X[train_indices & (groups != group)][features],
                y[train_indices & (groups != group)],
            )

        y_pred = pipe.predict_proba(X[test_indices & (groups == group)][features])[:, 1]
        y_true = y[test_indices & (groups == group)]

        results.loc[groups == group, "y_pred"] = y_pred

        aucs.append(
            metrics.roc_auc_score(y_true, y_pred)
        )  # if np.unique(y_true).size == 2 else np.nan)

        if hasattr(pipe["predict"], "coef_"):
            coefs.loc[group, features] = pipe["predict"].coef_

    print(f"AUC: {meanstd_str(aucs)}")
    evaluate_performance(
        results["y_test"],
        results["y_pred"],
        results["id"],
        results["scenario"],
        print_df=True,
        print_csv=True,
        threshold=-1,
        name=f"{desc}_{eval_type}",
    )
    print()

    if store_files:
        filename = f"{desc}_train_{train_type}_eval_{eval_type}"
        results.to_pickle(f"output/{filename}.pkl")
        coefs.to_pickle(f"output/{filename}_coefs.pkl")
