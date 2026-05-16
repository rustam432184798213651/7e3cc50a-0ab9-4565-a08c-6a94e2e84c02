model_name = "KNN"
# Available values:
# "Logistic Regression"
# "Decision Tree"
# "Random Forest"
# "KNN"
# "XGBoost"
# "Neural Network"

import os
import json
import random
import time
from collections import defaultdict
from heapq import heappop, heappush
from typing import Mapping, Sequence

RANDOM_STATE = 42

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

# Тут устанавливаем 1 thread, потому что segmentation fault вылетает иначе
if model_name == 'Neural Network':
    # Ограничиваем число нативных потоков до импорта numpy/sklearn/torch.
    # При многократном обучении нейросети внутри cross-validation это снижает
    # риск segmentation fault из-за конфликтов OpenMP/BLAS/PyTorch threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import ks_2samp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    kagglehub = None
    KaggleDatasetAdapter = None


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from skorch import NeuralNetClassifier
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
except ImportError:
    torch = None
    nn = None
    F = None
    NeuralNetClassifier = None


import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# ============================================================
# Global settings
# ============================================================

KAGGLE_DATASET = "sjleshrac/airlines-customer-satisfaction"
KAGGLE_DATASET_FILE = "Invistico_Airline.csv"
KAGGLE_DATASET_DIR = "datasets/sjleshrac_airlines_customer_satisfaction"

PASSENGER_SURVEY_PATH = "balanced_passenger_survey_dataset/passenger_survey_balanced.csv"
PASSENGER_SURVEY_SHEET = "Q4 2025"
PASSENGER_SURVEY_HEADER_ROW = 2

TRAIN_PATH = PASSENGER_SURVEY_PATH
TEST_PATH = None

TARGET_COLUMN = "liked"
SOURCE_SATISFACTION_COLUMN = "OVERALL SATISFACTION"
NEGATIVE_CLASS = "not_liked"
POSITIVE_CLASS = "liked"

TARGET_MAP = {
    NEGATIVE_CLASS: 0,
    POSITIVE_CLASS: 1,
}

TARGET_NAMES = [
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
]

FEATURE_SELECTION_K = 22
FAIRNESS_MIN_GROUP_SIZE = 30

LEAKAGE_COLUMNS = [
    SOURCE_SATISFACTION_COLUMN,
    "REASON",
    "ADDITIONAL COMMENTS",
]

IDENTIFIER_COLUMNS = [
    "KEY",
]

CONVERT_TO_BOOL = [
    'HAS A DISABILITY',
    'USES ASSISTIVE DEVICE',
    'REQUESTED SPECIAL ASSISTANCE',
    'DISEMBARKATION METHOD USED',
    'USED PARKING?',
    'USED PARKING',
]

USELESS_COLUMNS = [
    'AIRPORT',
    'SURVEY START',
    'SURVEY END',
    'TERMINAL',
    'GATE',
    'AIRLINE',
    'FLIGHT',
    'CONNECTION', # COME BACK
    'AIRLINE SERVICE 2',
    'AIRLINE SERVICE'
]

RESULTS_DIR = model_name.lower().replace(' ', '_') + "_results_second_approach"
IMAGES_DIR = model_name.lower().replace(' ', '_') + "_images_second_approach"

F1_SCORER = make_scorer(f1_score, average="macro")

CV_SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro"),
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "precision_macro": make_scorer(
        precision_score,
        average="macro",
        zero_division=0,
    ),
    "recall_macro": make_scorer(
        recall_score,
        average="macro",
        zero_division=0,
    ),
}

NON_NESTED_MODELS = {"Random Forest", "XGBoost", "Neural Network"}
PERMUTATION_IMPORTANCE_REPEATS = 5
PERMUTATION_IMPORTANCE_MAX_SAMPLES = 5000
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


# ============================================================
# Small utility transformers
# ============================================================

class ToFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class FillNonApplicableNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X

        X = X.copy()
        applicability_cols = [
            col for col in X.columns
            if col.endswith("_is_applicable")
        ]

        for indicator_col in applicability_cols:
            base_col = indicator_col[: -len("_is_applicable")]
            if base_col not in X.columns:
                continue

            not_applicable_mask = X[indicator_col].eq(0)
            X.loc[not_applicable_mask, base_col] = 0

        return X


if nn is not None:
    class TabularMLP(nn.Module):
        def __init__(
            self,
            input_dim,
            output_dim,
            hidden1=128,
            hidden2=64,
            dropout1=0.15,
            dropout2=0.10,
        ):
            super().__init__()

            if input_dim is None:
                self.fc1 = nn.LazyLinear(hidden1)
            else:
                self.fc1 = nn.Linear(input_dim, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.out = nn.Linear(hidden2, output_dim)

            self.dropout1 = dropout1
            self.dropout2 = dropout2

        def forward(self, X):
            X = F.relu(self.fc1(X))
            X = F.dropout(X, p=self.dropout1, training=self.training)

            X = F.relu(self.fc2(X))
            X = F.dropout(X, p=self.dropout2, training=self.training)

            X = self.out(X)
            return X


# ============================================================
# Data preprocessing
# ============================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(col).strip() for col in df.columns]

    for col in ["Unnamed: 0", "id", *IDENTIFIER_COLUMNS, *USELESS_COLUMNS]:
        if col in df.columns:
            df = df.drop(columns=col)

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else str(value)
                if pd.notna(value)
                else np.nan
            )
        )

    if SOURCE_SATISFACTION_COLUMN in df.columns:
        satisfaction_score = pd.to_numeric(
            df[SOURCE_SATISFACTION_COLUMN],
            errors="coerce",
        )
        df[TARGET_COLUMN] = satisfaction_score.isin([4, 5]).astype(int)
    elif TARGET_COLUMN in df.columns:
        numeric_target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if numeric_target.notna().all() and set(numeric_target.unique()).issubset({0, 1}):
            df[TARGET_COLUMN] = numeric_target.astype(int)
        else:
            target = df[TARGET_COLUMN].astype("string").str.lower().str.strip()
            df[TARGET_COLUMN] = target.map(TARGET_MAP)
    else:
        raise ValueError(
            f"Neither {SOURCE_SATISFACTION_COLUMN!r} nor "
            f"{TARGET_COLUMN!r} was found."
        )

    if df[TARGET_COLUMN].isna().any():
        unknown_values = sorted(df.loc[df[TARGET_COLUMN].isna(), TARGET_COLUMN].unique())
        raise ValueError(f"Unknown target values: {unknown_values}")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    columns_to_drop = [
        col for col in LEAKAGE_COLUMNS
        if col in df.columns
    ]
    df = df.drop(columns=columns_to_drop)

    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


def ensure_kaggle_dataset_downloaded() -> str:
    if os.path.exists(TRAIN_PATH):
        return TRAIN_PATH

    if kagglehub is None or KaggleDatasetAdapter is None:
        raise ImportError(
            "kagglehub is required to download the Kaggle dataset. "
            "Install it with: pip install 'kagglehub[pandas-datasets]'"
        )

    os.makedirs(KAGGLE_DATASET_DIR, exist_ok=True)

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        KAGGLE_DATASET_FILE,
    )

    df.to_csv(TRAIN_PATH, index=False)

    return TRAIN_PATH


def load_passenger_survey_dataframe() -> pd.DataFrame:
    if not os.path.exists(PASSENGER_SURVEY_PATH):
        raise FileNotFoundError(
            f"Passenger survey dataset was not found: {PASSENGER_SURVEY_PATH}"
        )

    if PASSENGER_SURVEY_PATH.lower().endswith(".csv"):
        return pd.read_csv(
            PASSENGER_SURVEY_PATH,
            encoding="utf-8-sig",
            keep_default_na=False,
            na_values=[""],
        )

    return pd.read_excel(
        PASSENGER_SURVEY_PATH,
        sheet_name=PASSENGER_SURVEY_SHEET,
        header=PASSENGER_SURVEY_HEADER_ROW,
    )


def load_data(frac: float | None = None):
    train_df = load_passenger_survey_dataframe()
    train_df = clean_dataframe(train_df)

    if frac:
        train_df = (
            train_df
            .groupby(TARGET_COLUMN, group_keys=False)
            .sample(frac=frac, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X_train_full, y_train_full = split_xy(train_df)

    if TEST_PATH and os.path.exists(TEST_PATH):
        test_df = pd.read_csv(TEST_PATH)
        test_df = clean_dataframe(test_df)
        if frac:
            test_df = (
                test_df
                .groupby(TARGET_COLUMN, group_keys=False)
                .sample(frac=frac, random_state=RANDOM_STATE)
                .reset_index(drop=True)
            )

        X_test, y_test = split_xy(test_df)

        X_test = X_test.reindex(columns=X_train_full.columns)

        X_train = X_train_full
        y_train = y_train_full
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=RANDOM_STATE,
        )

    return X_train, X_test, y_train, y_test


def load_combined_data(frac: float | None = None):
    train_df = load_passenger_survey_dataframe()
    train_df = clean_dataframe(train_df)

    dataframes = [train_df]

    if TEST_PATH and os.path.exists(TEST_PATH):
        test_df = pd.read_csv(TEST_PATH)
        test_df = clean_dataframe(test_df)
        dataframes.append(test_df)

    full_df = pd.concat(dataframes, ignore_index=True)

    if frac:
        full_df = (
            full_df
            .groupby(TARGET_COLUMN, group_keys=False)
            .sample(frac=frac, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X, y = split_xy(full_df)

    return X, y


# ============================================================
# Model configuration
# ============================================================

def normalize_model_name(name: str) -> str:
    aliases = {
        "lr": "Logistic Regression",
        "logreg": "Logistic Regression",
        "logistic regression": "Logistic Regression",

        "dt": "Decision Tree",
        "decision tree": "Decision Tree",

        "rf": "Random Forest",
        "random forest": "Random Forest",

        "knn": "KNN",

        "xgb": "XGBoost",
        "xgboost": "XGBoost",

        "nn": "Neural Network",
        "neural network": "Neural Network",
    }

    key = name.strip().lower()
    return aliases.get(key, name)


def make_safe_name(name: str) -> str:
    return (
        normalize_model_name(name)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def get_model_config(name: str):
    name = normalize_model_name(name)

    if name == "Logistic Regression":
        return {
            "step_name": "model",
            "search_type": "grid",
            "estimator_factory": lambda: LogisticRegression(
                solver="lbfgs",
                l1_ratio=0,
                max_iter=1000,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        }

    if name == "Decision Tree":
        return {
            "step_name": "model",
            "search_type": "grid",
            "estimator_factory": lambda: DecisionTreeClassifier(
                class_weight=None,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__max_depth": [14, 16, 18, 20, 24],
                "model__min_samples_leaf": [10, 20, 30, 50],
                "model__min_samples_split": [100, 200, 300, 500],
            },
        }

    if name == "Random Forest":
        return {
            "step_name": "model",
            "search_type": "randomized",
            "n_iter": 50,
            "estimator_factory": lambda: RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [500, 700, 900],
                "model__max_depth": [6, 8, 10, 12, 14, 16],
                "model__min_samples_split": [10, 20, 50, 100],
                "model__min_samples_leaf": [5, 10, 15, 20, 30],
                "model__max_features": [0.2, 0.3, 0.4, "sqrt"],
                "model__bootstrap": [True],
                "model__class_weight": [None],
            },
        }

    if name == "KNN":
        return {
            "step_name": "model",
            "search_type": "grid",
            "estimator_factory": lambda: KNeighborsClassifier(),
            "param_grid": {
                "model__n_neighbors": [31, 41, 51, 61, 81, 101],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan"],
            },
        }

    if name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

        return {
            "step_name": "model",
            "search_type": "randomized",
            "n_iter": 120,
            "estimator_factory": lambda: XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=RANDOM_STATE,
                n_jobs=4,
                verbosity=0,
            ),
            "param_grid": {
                "model__n_estimators": [600, 800, 1000],
                "model__max_depth": [5, 7, 9],
                "model__learning_rate": [0.015, 0.03, 0.05],
                "model__subsample": [0.9, 1.0],
                "model__colsample_bytree": [0.4, 0.6, 0.8],
                "model__reg_lambda": [0.3, 1, 3],
                "model__reg_alpha": [0, 0.1, 0.3],
            },
        }

    if name == "Neural Network":
        if NeuralNetClassifier is None or torch is None:
            raise ImportError("torch and skorch are required. Install them with: pip install torch skorch")

        return {
            "step_name": "model",
            "search_type": "randomized",
            "n_iter": 20,
            "estimator_factory": lambda: NeuralNetClassifier(
                module=TabularMLP,
                module__input_dim=None,
                module__output_dim=2,
                max_epochs=80,
                lr=0.001,
                batch_size=2048,
                optimizer=torch.optim.Adam,
                optimizer__weight_decay=1e-4,
                criterion=nn.CrossEntropyLoss,
                iterator_train__shuffle=True,
                iterator_train__num_workers=0,
                iterator_valid__num_workers=0,
                train_split=False,
                verbose=0,
                device="cuda" if torch.cuda.is_available() else "cpu",
            ),
            "param_grid": {
                "model__module__hidden1": [32, 64, 96],
                "model__module__hidden2": [16, 32, 64],
                "model__module__dropout1": [0.25, 0.30, 0.35, 0.40],
                "model__module__dropout2": [0.00, 0.05, 0.10],
                "model__lr": [0.0001, 0.0002, 0.0003],
                "model__batch_size": [1024, 2048, 4096],
                "model__optimizer__weight_decay": [0.0005, 0.001, 0.002],
            },
        }

    raise ValueError(f"Unknown model_name: {name!r}")


def build_pipeline(model_name: str, preprocessor: ColumnTransformer) -> Pipeline:
    config = get_model_config(model_name)

    steps = [
        ("fill_not_applicable_numeric", FillNonApplicableNumeric()),
        ("prep", preprocessor),
    ]

    if normalize_model_name(model_name) == "Neural Network":
        steps.append(("to_float32", ToFloat32()))

    steps.append((config["step_name"], config["estimator_factory"]()))

    return Pipeline(steps)


def build_search(model_name: str, pipeline: Pipeline, cv_strategy):
    config = get_model_config(model_name)
    normalized_name = normalize_model_name(model_name)

    if config["search_type"] == "randomized":
        n_jobs = 1 if normalized_name in {"Random Forest", "Neural Network", "XGBoost"} else 2

        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["param_grid"],
            n_iter=config.get("n_iter", 12),
            scoring=CV_SCORING,
            cv=cv_strategy,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            refit="f1_macro",
            verbose=0,
        )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=config["param_grid"],
        scoring=CV_SCORING,
        cv=cv_strategy,
        n_jobs=2,
        refit="f1_macro",
        verbose=0,
    )


# ============================================================
# CV helpers
# ============================================================

def make_nested_cv():
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=123,
    )

    return outer_cv, inner_cv


def make_curve_cv(model_name):
    normalized_name = normalize_model_name(model_name)

    if normalized_name in NON_NESTED_MODELS:
        return StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def bootstrap_metric_confidence_intervals(
    y_true,
    y_pred,
    n_bootstrap=1000,
    confidence_level=0.95,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Cannot build bootstrap intervals for an empty test set.")

    rng = np.random.default_rng(RANDOM_STATE)

    metric_functions = {
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "f1_weighted": lambda yt, yp: f1_score(
            yt,
            yp,
            average="weighted",
            labels=[0, 1],
            zero_division=0,
        ),
        "f1_macro": lambda yt, yp: f1_score(
            yt,
            yp,
            average="macro",
            labels=[0, 1],
            zero_division=0,
        ),
        "precision_weighted": lambda yt, yp: precision_score(
            yt,
            yp,
            average="weighted",
            labels=[0, 1],
            zero_division=0,
        ),
        "recall_weighted": lambda yt, yp: recall_score(
            yt,
            yp,
            average="weighted",
            labels=[0, 1],
            zero_division=0,
        ),
        "sensitivity": lambda yt, yp: recall_score(
            yt,
            yp,
            pos_label=1,
            zero_division=0,
        ),
        "specificity": lambda yt, yp: recall_score(
            yt,
            yp,
            pos_label=0,
            zero_division=0,
        ),
    }

    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    rows = []

    for metric_name, metric_func in metric_functions.items():
        point_estimate = metric_func(y_true, y_pred)
        bootstrap_values = []
        skipped_samples = 0

        for _ in range(n_bootstrap):
            sample_indices = rng.integers(0, len(y_true), size=len(y_true))
            sample_y_true = y_true[sample_indices]
            sample_y_pred = y_pred[sample_indices]

            # balanced_accuracy and macro metrics are not meaningful when a
            # bootstrap sample accidentally contains only one target class.
            if len(np.unique(sample_y_true)) < 2:
                skipped_samples += 1
                continue

            bootstrap_values.append(metric_func(sample_y_true, sample_y_pred))

        if bootstrap_values:
            bootstrap_values = np.asarray(bootstrap_values)
            bootstrap_mean = float(bootstrap_values.mean())
            bootstrap_std = float(bootstrap_values.std(ddof=1))
            ci_lower = float(np.percentile(bootstrap_values, lower_percentile))
            ci_upper = float(np.percentile(bootstrap_values, upper_percentile))
            n_bootstrap_used = len(bootstrap_values)
        else:
            bootstrap_mean = np.nan
            bootstrap_std = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            n_bootstrap_used = 0

        rows.append({
            "metric": metric_name,
            "point_estimate": point_estimate,
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_std": bootstrap_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "n_bootstrap_requested": n_bootstrap,
            "n_bootstrap_used": n_bootstrap_used,
            "n_bootstrap_skipped": skipped_samples,
        })

    return pd.DataFrame(rows)


def dump_metrics_and_confusion_matrix(model_name, y_true, y_pred):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "model": model_name,
        **metrics,
    }])

    metrics_path = os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    bootstrap_df = bootstrap_metric_confidence_intervals(
        y_true=y_true,
        y_pred=y_pred,
    )
    bootstrap_path = os.path.join(
        RESULTS_DIR,
        f"{make_safe_name(model_name)}_metric_confidence_intervals.csv",
    )
    bootstrap_df.to_csv(bootstrap_path, index=False, encoding="utf-8-sig")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).T
    report_path = os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_classification_report.csv")
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{NEGATIVE_CLASS}", f"true_{POSITIVE_CLASS}"],
        columns=[f"pred_{NEGATIVE_CLASS}", f"pred_{POSITIVE_CLASS}"],
    )

    cm_path = os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], [f"pred {NEGATIVE_CLASS}", f"pred {POSITIVE_CLASS}"], rotation=25, ha="right")
    plt.yticks([0, 1], [f"true {NEGATIVE_CLASS}", f"true {POSITIVE_CLASS}"])
    plt.title(f"Confusion Matrix: {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"{make_safe_name(model_name)}_confusion_matrix.png"), dpi=300)
    plt.close()

    return metrics


# ============================================================
# Research reports
# ============================================================

def dump_dataset_summary(X: pd.DataFrame, y: pd.Series):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    numeric_cols = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    target_counts = y.value_counts().sort_index()
    target_shares = y.value_counts(normalize=True).sort_index()

    feature_rows = []
    for col in X.columns:
        is_numeric = col in numeric_cols
        non_missing = X[col].dropna()

        row = {
            "feature": col,
            "dtype": str(X[col].dtype),
            "feature_type": "numeric" if is_numeric else "categorical",
            "missing_count": int(X[col].isna().sum()),
            "missing_rate": float(X[col].isna().mean()),
            "n_unique": int(X[col].nunique(dropna=True)),
            "is_applicability_indicator": col.endswith("_is_applicable"),
        }

        if is_numeric:
            row.update({
                "mean": float(non_missing.mean()) if len(non_missing) else np.nan,
                "std": float(non_missing.std()) if len(non_missing) else np.nan,
                "min": float(non_missing.min()) if len(non_missing) else np.nan,
                "median": float(non_missing.median()) if len(non_missing) else np.nan,
                "max": float(non_missing.max()) if len(non_missing) else np.nan,
                "top_value": np.nan,
                "top_value_share": np.nan,
            })
        else:
            value_counts = X[col].value_counts(normalize=True, dropna=True)
            row.update({
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "median": np.nan,
                "max": np.nan,
                "top_value": value_counts.index[0] if len(value_counts) else np.nan,
                "top_value_share": float(value_counts.iloc[0]) if len(value_counts) else np.nan,
            })

        feature_rows.append(row)

    feature_summary_df = pd.DataFrame(feature_rows).sort_values(
        by=["feature_type", "n_unique"],
        ascending=[True, False],
    )

    feature_summary_path = os.path.join(RESULTS_DIR, "dataset_feature_summary.csv")
    feature_summary_df.to_csv(
        feature_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    top_categorical_cardinality = (
        feature_summary_df[feature_summary_df["feature_type"] == "categorical"]
        .sort_values("n_unique", ascending=False)
        .head(20)
    )
    top_categorical_cardinality.to_csv(
        os.path.join(RESULTS_DIR, "dataset_top_categorical_cardinality.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "n_applicability_indicators": int(
            sum(col.endswith("_is_applicable") for col in X.columns)
        ),
        "target_counts": {
            str(k): int(v) for k, v in target_counts.to_dict().items()
        },
        "target_shares": {
            str(k): float(v) for k, v in target_shares.to_dict().items()
        },
        "total_missing_values": int(X.isna().sum().sum()),
        "features_with_missing_values": int((X.isna().sum() > 0).sum()),
        "max_missing_rate": float(X.isna().mean().max()),
        "feature_summary_csv": feature_summary_path,
    }

    summary_path = os.path.join(RESULTS_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, feature_summary_df


def get_positive_class_scores(fitted_model, X):
    if hasattr(fitted_model, "predict_proba"):
        proba = fitted_model.predict_proba(X)
        classes = np.asarray(getattr(fitted_model, "classes_", [0, 1]))
        positive_index = int(np.where(classes == 1)[0][0])
        return proba[:, positive_index], "predict_proba"

    if hasattr(fitted_model, "decision_function"):
        scores = fitted_model.decision_function(X)
        if np.ndim(scores) == 2:
            classes = np.asarray(getattr(fitted_model, "classes_", [0, 1]))
            positive_index = int(np.where(classes == 1)[0][0])
            scores = scores[:, positive_index]
        return np.asarray(scores), "decision_function"

    raise AttributeError(
        "The fitted model does not provide predict_proba or decision_function."
    )


def dump_permutation_importance(model_name, fitted_model, X_holdout, y_holdout):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    normalized_name = normalize_model_name(model_name)
    max_samples = min(PERMUTATION_IMPORTANCE_MAX_SAMPLES, len(X_holdout))

    if max_samples < len(X_holdout):
        X_eval = X_holdout.sample(
            n=max_samples,
            random_state=RANDOM_STATE,
        )
        y_eval = y_holdout.loc[X_eval.index]
    else:
        X_eval = X_holdout.copy()
        y_eval = y_holdout.copy()

    base_prediction = fitted_model.predict(X_eval)
    base_score = f1_score(
        y_eval,
        base_prediction,
        average="macro",
        zero_division=0,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    feature_rows = []
    features_to_permute = [
        col for col in X_eval.columns
        if not col.endswith("_is_applicable")
        and not col.endswith("is available")
    ]

    for feature in features_to_permute:
        repeat_importances = []

        for _ in range(PERMUTATION_IMPORTANCE_REPEATS):
            X_permuted = X_eval.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())

            permuted_prediction = fitted_model.predict(X_permuted)
            permuted_score = f1_score(
                y_eval,
                permuted_prediction,
                average="macro",
                zero_division=0,
            )

            repeat_importances.append(base_score - permuted_score)

        repeat_importances = np.asarray(repeat_importances)
        feature_rows.append({
            "feature": feature,
            "importance_mean": float(repeat_importances.mean()),
            "importance_std": float(repeat_importances.std(ddof=1))
            if len(repeat_importances) > 1 else 0.0,
            "baseline_f1_macro": float(base_score),
            "n_repeats": PERMUTATION_IMPORTANCE_REPEATS,
            "max_samples": max_samples,
            "excluded_is_applicable_features": True,
            "excluded_is_available_features": True,
        })

    importance_df = pd.DataFrame(feature_rows).sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(0, "rank", importance_df.index + 1)

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{make_safe_name(normalized_name)}_permutation_importance.csv",
    )
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    top_df = importance_df.head(25).sort_values("importance_mean")

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_df["feature"].astype(str),
        top_df["importance_mean"],
        xerr=top_df["importance_std"],
    )
    plt.xlabel("Decrease in F1-macro after permutation")
    plt.title(f"Permutation importance: {normalized_name}")
    plt.grid(axis="x")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            IMAGES_DIR,
            f"{make_safe_name(normalized_name)}_permutation_importance.png",
        ),
        dpi=300,
    )
    plt.close()

    return importance_df


def dump_threshold_analysis(model_name, fitted_model, X_holdout, y_holdout):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    normalized_name = normalize_model_name(model_name)
    scores, score_source = get_positive_class_scores(fitted_model, X_holdout)

    if score_source == "predict_proba":
        thresholds = THRESHOLD_GRID
    else:
        raise Exception('This model does not have predict_proba which is unexpected.')

    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "score_source": score_source,
            "accuracy": accuracy_score(y_holdout, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_holdout, y_pred),
            "f1_macro": f1_score(y_holdout, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_holdout, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(
                y_holdout,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_holdout,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "sensitivity": recall_score(
                y_holdout,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
            "specificity": recall_score(
                y_holdout,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),
            "precision_not_liked": precision_score(
                y_holdout,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),
            "recall_not_liked": recall_score(
                y_holdout,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),
            "precision_liked": precision_score(
                y_holdout,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
            "recall_liked": recall_score(
                y_holdout,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
            "pred_positive_rate": float((y_pred == 1).mean()),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df = threshold_df.sort_values("threshold").reset_index(drop=True)

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{make_safe_name(normalized_name)}_threshold_analysis.csv",
    )
    threshold_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    best_threshold_df = (
        threshold_df
        .sort_values(["f1_macro", "balanced_accuracy"], ascending=False)
        .head(1)
    )
    best_threshold_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{make_safe_name(normalized_name)}_best_threshold.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 6))
    plt.plot(threshold_df["threshold"], threshold_df["f1_macro"], marker="o", label="F1-macro")
    plt.plot(threshold_df["threshold"], threshold_df["precision_macro"], marker="o", label="Precision-macro")
    plt.plot(threshold_df["threshold"], threshold_df["recall_macro"], marker="o", label="Recall-macro")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(f"Threshold analysis: {normalized_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            IMAGES_DIR,
            f"{make_safe_name(normalized_name)}_threshold_analysis.png",
        ),
        dpi=300,
    )
    plt.close()

    return threshold_df


def dump_roc_auc_curve(model_name, fitted_model, X_holdout, y_holdout):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    normalized_name = normalize_model_name(model_name)
    scores, score_source = get_positive_class_scores(fitted_model, X_holdout)

    if len(np.unique(y_holdout)) < 2:
        raise ValueError("ROC AUC requires both target classes in y_holdout.")

    fpr, tpr, thresholds = roc_curve(y_holdout, scores, pos_label=1)
    roc_auc = roc_auc_score(y_holdout, scores)

    roc_df = pd.DataFrame({
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": thresholds,
        "score_source": score_source,
    })

    roc_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_roc_curve.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    roc_metrics_df = pd.DataFrame([{
        "model": normalized_name,
        "roc_auc": roc_auc,
        "score_source": score_source,
        "n_holdout_objects": len(y_holdout),
    }])
    roc_metrics_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_roc_auc_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve: {normalized_name}")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGES_DIR, f"{make_safe_name(normalized_name)}_roc_curve.png"),
        dpi=300,
    )
    plt.close()

    return roc_metrics_df, roc_df


# ============================================================
# Hyperparameter tuning
# ============================================================

def params_to_key(params: dict):
    return tuple(sorted(params.items()))


def choose_final_params(best_params_by_fold, outer_scores):
    grouped = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "params": None})

    for params, score in zip(best_params_by_fold, outer_scores):
        key = params_to_key(params)

        grouped[key]["count"] += 1
        grouped[key]["score_sum"] += score
        grouped[key]["params"] = params

    rows = []

    for item in grouped.values():
        rows.append({
            "params": item["params"],
            "count": item["count"],
            "mean_outer_f1": item["score_sum"] / item["count"],
        })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["count", "mean_outer_f1"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_params = summary.loc[0, "params"]
    best_score = summary.loc[0, "mean_outer_f1"]

    return best_params, best_score, summary


def run_nested_cv(model_name, X_train, y_train, preprocessor):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    outer_cv, inner_cv = make_nested_cv()

    best_params_by_fold = []
    outer_scores = []
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
        print("=" * 80)
        print(f"{model_name}: outer fold {fold_id}")

        X_outer_train = X_train.iloc[train_idx]
        X_outer_test = X_train.iloc[test_idx]

        y_outer_train = y_train.iloc[train_idx]
        y_outer_test = y_train.iloc[test_idx]

        pipeline = build_pipeline(model_name, preprocessor)
        search = build_search(model_name, pipeline, inner_cv)

        search.fit(X_outer_train, y_outer_train)

        y_pred = search.predict(X_outer_test)
        fold_score = f1_score(y_outer_test, y_pred, average="macro")

        best_params_by_fold.append(search.best_params_)
        outer_scores.append(fold_score)

        fold_rows.append({
            "fold": fold_id,
            "outer_f1_macro": fold_score,
            "best_inner_score": search.best_score_,
            "best_params": str(search.best_params_),
        })

        print(f"Best params: {search.best_params_}")
        print(f"Inner CV best F1-macro: {search.best_score_:.6f}")
        print(f"Outer F1-macro: {fold_score:.6f}")

    nested_df = pd.DataFrame(fold_rows)

    nested_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_nested_cv_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    best_params, best_score, params_summary = choose_final_params(
        best_params_by_fold,
        outer_scores,
    )

    params_summary.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_nested_cv_params_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print(f"Final selected params for {model_name}:")
    print(best_params)
    print(f"Selected nested score: {best_score:.6f}")

    return best_params, best_score, nested_df, params_summary


def run_simple_cv_search(model_name, X_train, y_train, preprocessor):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(model_name, preprocessor)
    search = build_search(model_name, pipeline, cv_strategy)

    print("=" * 80)
    print(f"{model_name}: simple CV hyperparameter search")
    print(f"CV splits: {cv_strategy.get_n_splits()}")

    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_score = search.best_score_
    best_estimator = search.best_estimator_

    print(f"Best params: {best_params}")
    print(f"Best CV F1-macro: {best_score:.6f}")

    cv_results_df = pd.DataFrame(search.cv_results_)

    cv_results_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_cv_search_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    best_result = cv_results_df.loc[search.best_index_]

    summary_row = {
        "params": best_params,
        "mean_cv_f1": best_score,
        "cv_splits": cv_strategy.get_n_splits(),
        "search_type": get_model_config(model_name)["search_type"],
    }

    for metric_name in CV_SCORING:
        summary_row[f"mean_cv_{metric_name}"] = best_result[f"mean_test_{metric_name}"]
        summary_row[f"std_cv_{metric_name}"] = best_result[f"std_test_{metric_name}"]

    summary_df = pd.DataFrame([summary_row])

    summary_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_cv_search_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return best_params, best_score, cv_results_df, summary_df, best_estimator


def run_hyperparameter_selection(model_name, X_train, y_train, preprocessor):
    normalized_name = normalize_model_name(model_name)

    return run_simple_cv_search(
        model_name=normalized_name,
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
    )


# ============================================================
# Final model training and performance
# ============================================================

def fit_final_model(model_name, best_params, X_train, y_train, preprocessor):
    pipeline = build_pipeline(model_name, preprocessor)
    pipeline.set_params(**best_params)

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    return pipeline, fit_time


def dump_cv_performance(model_name, params_summary, n_objects):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    row = {
        "model": model_name,
        "evaluation_type": "cross_validation",
        "n_objects": n_objects,
    }

    if len(params_summary) > 0:
        row.update(params_summary.iloc[0].to_dict())

    perf_path = os.path.join(RESULTS_DIR, "models_performance.csv")
    row_df = pd.DataFrame([row])

    if os.path.exists(perf_path):
        old_df = pd.read_csv(perf_path)
        old_df = old_df[old_df["model"] != model_name]
        row_df = pd.concat([old_df, row_df], ignore_index=True)

    row_df.to_csv(perf_path, index=False, encoding="utf-8-sig")

    return row


# ============================================================
# Feature selection table
# ============================================================

def dump_feature_selection_table(model_name, prep, selector):
    # ДОРАБОТАТЬ
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(selector.scores_))])

    support = selector.get_support()

    all_scores = pd.DataFrame({
        "feature": feature_names,
        "score": selector.scores_,
        "p_value": selector.pvalues_,
        "selected": support,
    })

    all_scores = all_scores.sort_values(
        by="score",
        ascending=False,
    ).reset_index(drop=True)

    all_scores.insert(0, "rank", all_scores.index + 1)

    path = os.path.join(RESULTS_DIR, f"{make_safe_name(model_name)}_feature_selection_scores.csv")
    all_scores.to_csv(path, index=False, encoding="utf-8-sig")

    return all_scores


# ============================================================
# Validation curves
# ============================================================

def make_hashable_params(params: dict):
    def convert(value):
        if isinstance(value, dict):
            return tuple(sorted((k, convert(v)) for k, v in value.items()))

        if isinstance(value, list):
            return tuple(convert(x) for x in value)

        if isinstance(value, set):
            return tuple(sorted(convert(x) for x in value))

        if isinstance(value, np.generic):
            return value.item()

        return value

    return tuple(sorted((key, convert(value)) for key, value in params.items()))


def complete_best_params(param_grid: dict, best_params: dict) -> dict:
    completed_params = best_params.copy()

    for param_name, param_values in param_grid.items():
        if len(param_values) == 1 and param_name not in completed_params:
            completed_params[param_name] = param_values[0]

    missing_params = []

    for param_name, param_values in param_grid.items():
        if len(param_values) > 1 and param_name not in completed_params:
            missing_params.append(param_name)

    if missing_params:
        raise ValueError(
            "best_params does not contain all parameters required for validation curves: "
            f"{missing_params}"
        )

    return completed_params


# ============================================================
# Learning curve
# ============================================================

def dump_learning_curve(model_name, final_params, X, y, preprocessor):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    normalized_name = normalize_model_name(model_name)

    pipeline = build_pipeline(normalized_name, preprocessor)
    pipeline.set_params(**final_params)

    train_sizes = np.linspace(0.1, 1.0, 10)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_jobs = 1 if model_name == 'Neural Network' else -1

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=F1_SCORER,
        n_jobs=n_jobs,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    curve_df = pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_mean_f1_macro": train_mean,
        "train_std_f1_macro": train_std,
        "valid_mean_f1_macro": valid_mean,
        "valid_std_f1_macro": valid_std,
    })

    curve_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_learning_curve.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes_abs,
        valid_mean,
        marker="o",
        label="Cross-validation score",
    )

    plt.xlabel("Number of training samples")
    plt.ylabel("F1-macro")
    plt.title(f"Learning curve: {normalized_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(IMAGES_DIR, f"{make_safe_name(normalized_name)}_learning_curve.png"),
        dpi=300,
    )

    plt.close()

    return curve_df


# ============================================================
# Error and fairness analysis
# ============================================================

def run_error_and_fairness_analysis(
    model_name,
    fitted_model,
    X_test,
    y_test,
    sensitive_cols=None,
):
    # Создаем папки для табличных результатов и графиков, если их еще нет.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Приводим название модели к единому виду, чтобы использовать его
    # в именах файлов и заголовках графиков.
    normalized_name = normalize_model_name(model_name)

    # Если пользователь не передал список чувствительных признаков,
    # берем несколько категориальных признаков, по которым удобно проверить,
    # одинаково ли модель ошибается для разных групп пассажиров.
    if sensitive_cols is None:
        sensitive_cols = [
            "gender",
            "age_group",
            "nationality",
            "flight_type",
            "process",
            "trip_purpose",
        ]

    # Получаем предсказания уже обученной модели на тестовой выборке.
    y_pred = fitted_model.predict(X_test)

    # Собираем общий датафрейм для анализа: исходные признаки, правильный
    # ответ, предсказание модели и флаг ошибки по каждой строке.
    analysis_df = X_test.copy()
    analysis_df["y_true"] = y_test.values if hasattr(y_test, "values") else y_test
    analysis_df["y_pred"] = y_pred
    analysis_df["is_error"] = analysis_df["y_true"] != analysis_df["y_pred"]

    # Создаем возрастные группы только для анализа. В модель эти столбцы
    # не передаются, поэтому они не меняют предсказания.
    if "Age" in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df["Age"]):
        analysis_df["Age group"] = pd.cut(
            analysis_df["Age"],
            bins=[0, 18, 30, 45, 60, np.inf],
            labels=["0-18", "19-30", "31-45", "46-60", "61+"],
            include_lowest=True,
        )

    if "age_group" in analysis_df.columns:
        analysis_df["age_group"] = analysis_df["age_group"].astype("string").fillna("Unknown")

    # Делим дальность перелета на понятные интервалы: короткие, средние,
    # длинные и очень длинные рейсы.
    if "Flight Distance" in analysis_df.columns:
        analysis_df["Flight distance group"] = pd.cut(
            analysis_df["Flight Distance"],
            bins=[0, 500, 1500, 3000, np.inf],
            labels=["short", "medium", "long", "very_long"],
            include_lowest=True,
        )

    # Оставляем только те чувствительные признаки, которые реально есть
    # в таблице анализа. Это защищает функцию от падения при отсутствии
    # какого-либо столбца.
    sensitive_cols = [
        col for col in sensitive_cols
        if col in analysis_df.columns
    ]

    # Определяем тип ошибки для каждой записи:
    # false_positive - модель предсказала удовлетворенного пассажира вместо
    # нейтрального/недовольного, false_negative - наоборот.
    def get_error_type(row):
        if row["y_true"] == row["y_pred"]:
            return "correct"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "false_positive"
        if row["y_true"] == 1 and row["y_pred"] == 0:
            return "false_negative"
        return "other_error"

    analysis_df["error_type"] = analysis_df.apply(get_error_type, axis=1)

    # Если модель умеет возвращать вероятности классов, добавляем вероятность
    # класса "satisfied" и общую уверенность модели в выбранном классе.
    # Для моделей без predict_proba этот блок просто пропускается.
    try:
        proba = fitted_model.predict_proba(X_test)
        analysis_df["proba_satisfied"] = proba[:, 1]
        analysis_df["model_confidence"] = np.max(proba, axis=1)
    except Exception:
        pass

    fairness_rows = []

    # Считаем качество отдельно внутри каждой группы каждого чувствительного
    # признака. Например, отдельно для Gender=Male и Gender=Female.
    for col in sensitive_cols:
        for group_value, group_df in analysis_df.groupby(col, dropna=False, observed=True):
            y_true_g = group_df["y_true"]
            y_pred_g = group_df["y_pred"]

            # В строку результата попадают размер группы, доля ошибок,
            # F1/precision/recall и доли положительных классов. Эти значения
            # помогают увидеть, в каких группах модель работает хуже.
            fairness_rows.append({
                "attribute": col,
                "group": group_value,
                "n_objects": len(group_df),
                "error_rate": group_df["is_error"].mean(),
                "f1_weighted": f1_score(y_true_g, y_pred_g, average="weighted", zero_division=0),
                "f1_macro": f1_score(y_true_g, y_pred_g, average="macro", zero_division=0),
                "precision_positive": precision_score(y_true_g, y_pred_g, pos_label=1, zero_division=0),
                "recall_positive": recall_score(y_true_g, y_pred_g, pos_label=1, zero_division=0),
                "pred_positive_rate": (y_pred_g == 1).mean(),
                "true_positive_rate_base": (y_true_g == 1).mean(),
            })

    fairness_df = pd.DataFrame(fairness_rows)

    disparity_rows = []

    # Для каждого чувствительного признака считаем разрыв между лучшей
    # и худшей достаточно крупной группой. Малые группы дают нестабильные
    # значения F1 0.0 или 1.0 и могут создавать ложный fairness gap.
    if len(fairness_df) > 0:
        for attr, attr_df in fairness_df.groupby("attribute"):
            eligible_df = attr_df[attr_df["n_objects"] >= FAIRNESS_MIN_GROUP_SIZE]
            if len(eligible_df) < 2:
                continue
            disparity_rows.append({
                "attribute": attr,
                "metric": "f1_macro",
                "min_group_size": FAIRNESS_MIN_GROUP_SIZE,
                "n_groups_total": len(attr_df),
                "n_groups_used": len(eligible_df),
                "f1_gap": eligible_df["f1_macro"].max() - eligible_df["f1_macro"].min(),
                "error_rate_gap": eligible_df["error_rate"].max() - eligible_df["error_rate"].min(),
                "recall_positive_gap": eligible_df["recall_positive"].max() - eligible_df["recall_positive"].min(),
            })

    disparity_df = pd.DataFrame(disparity_rows)

    # Сохраняем подробный анализ по объектам: где модель ошиблась,
    # какой тип ошибки был получен и, если доступно, с какой уверенностью.
    analysis_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_error_analysis.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Сохраняем метрики качества по каждой группе чувствительных признаков.
    fairness_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_fairness_by_group.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Сохраняем агрегированные разрывы качества между группами.
    disparity_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_fairness_gaps.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Отдельно строим график разрывов F1 между группами для каждого
    # чувствительного признака. Он показывает, где различия качества
    # наиболее заметны.
    if len(disparity_df) > 0:
        plt.figure(figsize=(8, 5))
        plt.bar(disparity_df["attribute"].astype(str), disparity_df["f1_gap"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("F1 gap")
        plt.title(f"Fairness F1 gaps: {normalized_name}")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(IMAGES_DIR, f"{make_safe_name(normalized_name)}_fairness_f1_gaps.png"),
            dpi=300,
        )
        plt.close()

    # Возвращаем три таблицы, чтобы их можно было дополнительно изучать
    # в коде без чтения сохраненных CSV-файлов.
    return analysis_df, fairness_df, disparity_df


# ============================================================
# Robustness check
# ============================================================

def add_numeric_noise(X_part, numeric_cols, train_std, noise_level, rng):
    X_noisy = X_part.copy()

    for col in numeric_cols:
        std = train_std.get(col, 0.0)

        if pd.isna(std) or std == 0:
            continue

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * std,
            size=len(X_noisy),
        )

        X_noisy[col] = X_noisy[col].astype(float) + noise

    return X_noisy


def add_missing_values(X_part, missing_rate, rng):
    X_missing = X_part.copy()

    rows, cols = X_missing.shape
    mask = rng.random((rows, cols)) < missing_rate

    for col_id, col in enumerate(X_missing.columns):
        X_missing.loc[mask[:, col_id], col] = np.nan

    return X_missing


def replace_categorical_values(X_part, categorical_cols, categories_by_col, replace_rate, rng):
    X_changed = X_part.copy()

    for col in categorical_cols:
        possible_values = categories_by_col.get(col, [])

        if len(possible_values) == 0:
            continue

        mask = rng.random(len(X_changed)) < replace_rate

        if mask.sum() == 0:
            continue

        X_changed.loc[mask, col] = rng.choice(
            possible_values,
            size=mask.sum(),
            replace=True,
        )

    return X_changed


def run_model_robustness_check(model_name, fitted_model, X_train, X_test, y_test):
    # Создаем папки, куда будут сохранены таблица с результатами
    # robustness-проверки и графики изменения качества.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Нормализованное имя модели используется в названиях файлов и графиков.
    normalized_name = normalize_model_name(model_name)

    # Фиксируем генератор случайных чисел, чтобы шум, пропуски и замены
    # воспроизводились одинаково при повторных запусках.
    rng = np.random.default_rng(RANDOM_STATE)

    # Разделяем признаки по типам, потому что для числовых и категориальных
    # колонок используются разные виды искусственного "повреждения" данных.
    numeric_cols = X_test.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_cols = X_test.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    # Стандартные отклонения считаются по обучающей выборке: они задают
    # масштаб шума для числовых признаков.
    train_std = X_train[numeric_cols].std(numeric_only=True).to_dict()

    # Для категориальных признаков запоминаем допустимые значения из train,
    # чтобы при замене не создавать несуществующие категории.
    categories_by_col = {
        col: X_train[col].dropna().unique()
        for col in categorical_cols
    }

    results = []

    # Сначала считаем качество на чистой тестовой выборке. Это базовый
    # уровень, с которым дальше сравниваются все искаженные версии данных.
    y_pred_clean = fitted_model.predict(X_test)

    baseline_f1_macro = f1_score(y_test, y_pred_clean, average="macro")

    results.append({
        "test_type": "baseline_clean",
        "level": 0.0,
        "f1_macro": baseline_f1_macro,
        "f1_weighted": f1_score(y_test, y_pred_clean, average="weighted"),
        "accuracy": accuracy_score(y_test, y_pred_clean),
    })

    # Проверяем устойчивость к небольшому случайному шуму в числовых колонках.
    # noise_level означает долю от стандартного отклонения признака.
    for noise_level in [0.01, 0.03, 0.05, 0.10]:
        X_noisy = add_numeric_noise(
            X_test,
            numeric_cols,
            train_std,
            noise_level,
            rng,
        )

        y_pred = fitted_model.predict(X_noisy)

        results.append({
            "test_type": "numeric_noise",
            "level": noise_level,
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        })

    # Проверяем, как модель переносит случайные пропуски во входных данных.
    # missing_rate задает вероятность заменить конкретную ячейку на NaN.
    for missing_rate in [0.01, 0.03, 0.05]:
        X_missing = add_missing_values(
            X_test,
            missing_rate,
            rng,
        )

        y_pred = fitted_model.predict(X_missing)

        results.append({
            "test_type": "missing_values",
            "level": missing_rate,
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        })

    # Проверяем чувствительность к ошибкам в категориальных признаках:
    # часть значений случайно заменяется на другие допустимые категории.
    for replace_rate in [0.01, 0.03, 0.05]:
        X_changed = replace_categorical_values(
            X_test,
            categorical_cols,
            categories_by_col,
            replace_rate,
            rng,
        )

        y_pred = fitted_model.predict(X_changed)

        results.append({
            "test_type": "categorical_replacement",
            "level": replace_rate,
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        })

    # Собираем все результаты в одну таблицу: baseline и все сценарии
    # искусственного ухудшения данных.
    robustness_df = pd.DataFrame(results)

    # Для каждого сценария считаем, насколько macro F1 упал относительно качества
    # на чистой тестовой выборке. Для baseline падение будет равно нулю.
    robustness_df["f1_macro_drop_vs_baseline"] = (
        baseline_f1_macro - robustness_df["f1_macro"]
    )

    # Сохраняем таблицу, чтобы потом можно было использовать ее в отчете
    # или сравнить устойчивость разных моделей.
    robustness_df.to_csv(
        os.path.join(RESULTS_DIR, f"{make_safe_name(normalized_name)}_robustness.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Строим общий график падения macro F1 относительно baseline для всех сценариев.
    # По нему удобно быстро увидеть, какой тип искажения сильнее всего
    # ухудшает качество модели.
    plt.figure(figsize=(9, 5))

    plot_df = robustness_df[robustness_df["test_type"] != "baseline_clean"].copy()
    plot_df["case"] = plot_df["test_type"] + "_" + plot_df["level"].astype(str)

    plt.bar(plot_df["case"], plot_df["f1_macro_drop_vs_baseline"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1 drop vs baseline")
    plt.title(f"Robustness macro F1 drop: {normalized_name}")
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        os.path.join(IMAGES_DIR, f"{make_safe_name(normalized_name)}_robustness_f1_drop.png"),
        dpi=300,
    )

    plt.close()

    # Возвращаем таблицу с результатами, чтобы ее можно было использовать
    # дальше в коде без чтения CSV-файла.
    return robustness_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    selected_model_name = normalize_model_name(model_name)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    X, y = load_combined_data()
    dump_dataset_summary(X, y)

    #  Здесь название функции train_test_split не совсем правильное
    #  X_cv и y_cv будет использованы для обучения и валидации с помощью кросс валидации.
    #  Причина почему мы отщепляем 10% от датасета, это нужно для fairness и error analysis + для robustness check
    X_cv, X_holdout, y_cv, y_holdout = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_cv)

    best_params, search_score, search_df, params_summary, final_model = run_hyperparameter_selection(
        selected_model_name,
        X_cv,
        y_cv,
        preprocessor,
    )

    performance_row = dump_cv_performance(
        selected_model_name,
        params_summary,
        len(X_cv),
    )

    dump_learning_curve(
        selected_model_name,
        best_params,
        X_cv,
        y_cv,
        preprocessor,
    )

    y_holdout_pred = final_model.predict(X_holdout)
    dump_metrics_and_confusion_matrix(
        selected_model_name,
        y_holdout,
        y_holdout_pred,
    )

    dump_permutation_importance(
        selected_model_name,
        final_model,
        X_holdout,
        y_holdout,
    )

    dump_threshold_analysis(
        selected_model_name,
        final_model,
        X_holdout,
        y_holdout,
    )

    dump_roc_auc_curve(
        selected_model_name,
        final_model,
        X_holdout,
        y_holdout,
    )

    run_error_and_fairness_analysis(
        selected_model_name,
        final_model,
        X_holdout,
        y_holdout,
    )

    run_model_robustness_check(
        selected_model_name,
        final_model,
        X_cv,
        X_holdout,
        y_holdout,
    )

    print("\nDone.")
    print("Model:", selected_model_name)
    print("Best params:", best_params)
    print("Search CV score:", search_score)
    print("Final CV performance:")
    print(performance_row)


# ============================================================
# Optional business utility
# ============================================================

def find_cheapest_way_to_make_customer_happy(
    current_state: Sequence,
    cost_to_change_state: Mapping[object, Mapping[object, list[tuple[object, float]]]],
    model,
):
    if isinstance(current_state, pd.Series):
        feature_names = list(current_state.index)
        start = tuple(current_state.values)
    else:
        feature_names = None
        start = tuple(current_state)

    if feature_names is None and hasattr(model, "named_steps") and "prep" in model.named_steps:
        prep = model.named_steps["prep"]

        if hasattr(prep, "feature_names_in_"):
            feature_names = list(prep.feature_names_in_)

    if feature_names is None:
        feature_names = list(cost_to_change_state.keys())

    if len(feature_names) != len(start):
        raise ValueError(
            "Размер current_state не совпадает с количеством признаков, "
            "которые ожидает final_model. Передайте current_state как "
            "pd.Series с теми же колонками, что были в X_train, или список "
            "значений в порядке этих колонок."
        )

    missing_cost_features = [
        feature_name
        for feature_name in feature_names
        if feature_name not in cost_to_change_state
    ]

    if missing_cost_features:
        raise ValueError(
            "В cost_to_change_state отсутствуют признаки: "
            f"{missing_cost_features}"
        )

    def make_model_input(state: tuple[int, ...]):
        return pd.DataFrame([state], columns=feature_names)

    def is_satisfied(state: tuple[int, ...]) -> bool:
        prediction = model.predict(make_model_input(state))[0]
        return prediction == 1

    if is_satisfied(start):
        return {
            "cost": 0,
            "path": [start],
        }

    distances = {start: 0}
    previous = {start: None}

    heap = []
    heappush(heap, (0, start))

    visited = set()

    while heap:
        current_cost, state = heappop(heap)

        if state in visited:
            continue

        visited.add(state)

        if is_satisfied(state):
            path = []
            current = state

            while current is not None:
                path.append(current)
                current = previous[current]

            path.reverse()

            return {
                "cost": current_cost,
                "path": path,
            }

        for index, feature_name in enumerate(feature_names):
            current_value = state[index]
            possible_changes = cost_to_change_state[feature_name].get(
                current_value,
                [],
            )

            for new_value, transition_cost in possible_changes:
                if state[index] == new_value:
                    continue

                new_state = list(state)
                new_state[index] = new_value
                new_state = tuple(new_state)

                new_cost = current_cost + transition_cost

                if new_state not in distances or new_cost < distances[new_state]:
                    distances[new_state] = new_cost
                    previous[new_state] = state
                    heappush(heap, (new_cost, new_state))

    return None
