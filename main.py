import glob
import os

import xgboost

try:  # Speedup for intel processors
    from sklearnex import patch_sklearn

    patch_sklearn()
except:
    pass

import numpy as np
import pandas as pd
from sklearn import (
    metrics,
    pipeline,
    preprocessing,
    linear_model,
    neural_network,
    ensemble,
)

from data_reader import read_data

from feature_selection import (
    calculate_cv_score,
    meanstd_str,
)

np.random.seed(0)

############# USER SETTING
WINDOW_SIZE = 60

STUDY_1_SUBJECTS = [
    105,
    106,
    107,
    108,
    111,
    113,
    114,
    115,
    116,
    117,
    119,
    120,
    121,
    123,
    124,
    125,
    127,
    129,
]
STUDY_2_SUBJECTS = [301, 302, 303, 304, 307, 310, 312, 313, 315]

STUDY_1_SUBJECTS.remove(119)  # no Empatica

LOSO_SUBJECTS = STUDY_1_SUBJECTS + STUDY_2_SUBJECTS
#############


def load_data(window_size_sec, reload_data=False):
    parquet_filename = f"./data/all_data_{window_size_sec:03d}.parquet"
    if reload_data or not os.path.exists(parquet_filename):
        DATA_FOLDER = "/headwind/lab-study/"

        print("Preprocessing physio data")
        subject_paths = sorted(
            glob.glob(DATA_FOLDER + "/*-V3") + glob.glob(DATA_FOLDER + "/*_3[01][0-9]")
        )
        X = read_data(
            subject_paths,
            ibi_threshold=0.5,
            data_source="garmin",
            window_length=window_size_sec,
        )
        X = X[X["subject_id"].isin(STUDY_1_SUBJECTS + STUDY_2_SUBJECTS)]
        X = X[X["phase"].isin([1, 2, 3])]
        # filter non-finite columns, some entropy cols might be nan
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(
            axis="columns", subset=X.index[~X["env"].isna()], how="any", inplace=True
        )
        X["train"] = X["test"] = True
        X.to_parquet(parquet_filename)
    else:
        X = pd.read_parquet(parquet_filename)

    X["pred_cgm_30"], X["pred_cgm_39"] = X["cgm"] < 3.0, X["cgm"] < 3.9
    X["y_30"], X["y_39"] = X["bg"] < 3.0, X["bg"] < 3.9

    return X


def get_pipelines():
    predict_steps = [
        ### Robustness Checks
        (
            "predict",
            linear_model.LogisticRegression(C=1e-3, class_weight="balanced"),
        ),  # Ridge
        (
            "predict",
            linear_model.LogisticRegression(
                C=1e-3, penalty="l1", solver="saga", class_weight="balanced"
            ),
        ),  # Lasso
        (
            "predict",
            linear_model.LogisticRegression(
                C=1e-3,
                l1_ratio=0.5,
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
            ),
        ),  # Elasticnet
        (
            "predict",
            xgboost.sklearn.XGBClassifier(
                n_estimators=10, min_samples_split=10, max_depth=3, verbosity=0
            ),
        ),  # XGBoost
        (
            "predict",
            neural_network.MLPClassifier(activation="logistic", max_iter=50),
        ),  # MLP
        (
            "predict",
            ensemble.GradientBoostingClassifier(
                n_estimators=10, min_samples_split=10, max_depth=3
            ),
        ),  # Gradient boosting
    ]

    for predict_step in predict_steps:
        yield pipeline.Pipeline(
            [("scale", preprocessing.StandardScaler()), predict_step]
        )


def run_evaluation(window_size_sec):
    X = load_data(window_size_sec, reload_data=False)
    X = X[X["phase"].isin([1, 3])]
    # X = X[X['train'] | X['test']]
    X["train"] = X["test"] = True

    def sens_spec(y_true, y_pred):
        return pd.Series(
            {
                "Sens": metrics.recall_score(y_true, y_pred, pos_label=1),
                "Spec": metrics.recall_score(y_true, y_pred, pos_label=0),
            }
        )

    print(
        "Moderate",
        X[X["subject_id"].isin(STUDY_1_SUBJECTS)]
        .groupby(["subject_id"])
        .apply(lambda x: sens_spec(x["y_30"], x["pred_cgm_30"]))
        .apply(meanstd_str),
    )
    print(
        "Mild",
        X[X["subject_id"].isin(STUDY_2_SUBJECTS)]
        .groupby(["subject_id"])
        .apply(lambda x: sens_spec(x["y_39"], x["pred_cgm_39"]))
        .apply(meanstd_str),
    )
    print(
        "Mixed",
        X[X["subject_id"].isin(LOSO_SUBJECTS)]
        .groupby(["subject_id"])
        .apply(lambda x: sens_spec(x["y_39"], x["pred_cgm_39"]))
        .apply(meanstd_str),
    )

    label_column = "y_39"

    # X = X[((X['phase'] == 1) & (~X[label_column])) | ((X['phase'] == 3) & (X[label_column]))]

    evaluate_indices_studies = [
        X["test"] & X["subject_id"].isin(STUDY_1_SUBJECTS),
        X["test"] & X["subject_id"].isin(STUDY_2_SUBJECTS),
        X["test"] & X["subject_id"].isin(LOSO_SUBJECTS),
    ]

    train_test_configs = [
        (
            X["train"] & X["subject_id"].isin(STUDY_1_SUBJECTS),
            X["test"] & X["subject_id"].isin(STUDY_1_SUBJECTS),
            evaluate_indices_studies,
        ),
        (
            X["train"] & X["subject_id"].isin(STUDY_2_SUBJECTS),
            X["test"] & X["subject_id"].isin(STUDY_2_SUBJECTS),
            evaluate_indices_studies,
        ),
        (
            X["train"] & X["subject_id"].isin(LOSO_SUBJECTS),
            X["test"] & X["subject_id"].isin(LOSO_SUBJECTS),
            evaluate_indices_studies,
        ),
    ]

    X_save = X.copy()
    for train_indices, test_indices, evaluate_indices in train_test_configs:
        all_evaluate_indices = pd.concat(evaluate_indices, axis=1).any(axis=1)
        X = X_save[train_indices | test_indices | all_evaluate_indices]
        train_indices = train_indices[
            train_indices | test_indices | all_evaluate_indices
        ]
        test_indices = test_indices[train_indices | test_indices | all_evaluate_indices]

    
        features = [
            "eda_phasic_median",
            "eda_phasic_pct_5",
            "eda_phasic_pct_95",
            "eda_tonic_median",
            "eda_tonic_pct_5",
            "eda_tonic_pct_95",
            "acc_l2_median",
            "acc_l2_pct_5",
            "acc_l2_pct_95",
            "hrv_sdnn",
            "hrv_rmssd",
            "hrv_total_power",
            "hrv_lf_hf_ratio",
            "hrv_cvnni",
        ]

        calculate_cv_score(
            X=X[features],
            y=X[label_column],
            groups=X["subject_id"],
            scenarios=X["env"],
            train_indices=train_indices,
            test_indices=test_indices,
            features=features,
            desc="ALL",
            bg=X["bg"],
            store_files=True,
        )

    return


if __name__ == "__main__":
    run_evaluation(window_size_sec=WINDOW_SIZE)
