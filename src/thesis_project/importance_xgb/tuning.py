import logging

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from thesis_project.importance_xgb import (
    config,
)

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    config_obj: config.AnalysisConfig,
    seeds,
    X: pd.DataFrame,
    y: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
):
    fixed_params = config.get_xgb_fixed_params(config_obj)
    params = {
        # Fixed parameters
        **fixed_params,
        "n_estimators": 3000,  # Intentionally high, controlled by early stopping
        "early_stopping_rounds": 100,
        # Tree Booster
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 13, step=2),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 1e2, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1e1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1e1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1, log=True),
    }

    # Parallelize over seeds
    results = Parallel(
        n_jobs=config_obj.n_jobs_seed,
        backend="loky",
    )(
        delayed(train)(
            config_obj=config_obj,
            X=X,
            y=y,
            params=params,
            splits=splits,
            seed=seed,
        )
        for seed in seeds
    )

    scores = []
    best_iterations = []
    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
    }

    for seed_scores, seed_best_iterations, seed_metrics in results:  # type: ignore
        scores.extend(seed_scores)
        best_iterations.extend(seed_best_iterations)

        for k in metrics:
            metrics[k].extend(seed_metrics[k])

    for k, values in metrics.items():
        trial.set_user_attr(k, values)

    trial.set_user_attr("scores", [float(v) for v in scores])
    trial.set_user_attr("best_iterations", [int(v) for v in best_iterations])

    return float(np.mean(scores))


def train(
    config_obj: config.AnalysisConfig,
    X,
    y,
    params,
    splits,
    seed: int,
):
    scores = []
    best_iterations = []
    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
    }

    for fold, (idx_train, idx_val) in enumerate(splits):
        X_train = X.iloc[idx_train]
        X_val = X.iloc[idx_val]

        y_train = y.iloc[idx_train]
        y_val = y.iloc[idx_val]

        model = XGBClassifier(random_state=seed, **params)
        model.fit(X_train, y_train.iloc[:, 0], eval_set=[(X_val, y_val.iloc[:, 0])], verbose=False)

        y_val_proba = model.predict_proba(X_val)
        y_val_pred = model.predict(X_val)

        best_iteration = model.best_iteration
        best_iterations.append(best_iteration)

        score = log_loss(y_val.iloc[:, 0], y_val_proba, labels=range(config_obj.n_quantile))
        scores.append(score)

        fold_metrics = compute_metrics(y_val.iloc[:, 0], y_val_pred)
        for k, v in fold_metrics.items():
            metrics[k].append(v)

    return scores, best_iterations, metrics


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def optuna_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial, config_obj):
    if trial.state == optuna.trial.TrialState.PRUNED:
        logger.info(
            "[%3d/%3d] PRUNED",
            trial.number + 1,
            config_obj.optuna_n_trials,
        )
        return

    logger.info(
        " "
        "trial %3d: "
        "value=%.5f, "
        "std_value=%.5f, "
        # "best_score=%.5f (trial %d) | "
        "acc=%.5f, "
        "bal_acc=%.5f, "
        "prec=%.5f, "
        "rec=%.5f, "
        "f1=%.5f, "
        "mcc=%.5f",
        trial.number,
        # config_obj.optuna_n_trials - 1,
        trial.value,
        np.mean(trial.user_attrs.get("scores", float("nan"))),
        # study.best_value,
        # study.best_trial.number,
        np.mean(trial.user_attrs.get("accuracy", float("nan"))),
        np.mean(trial.user_attrs.get("balanced_accuracy", float("nan"))),
        np.mean(trial.user_attrs.get("precision", float("nan"))),
        np.mean(trial.user_attrs.get("recall", float("nan"))),
        np.mean(trial.user_attrs.get("f1", float("nan"))),
        np.mean(trial.user_attrs.get("mcc", float("nan"))),
    )
