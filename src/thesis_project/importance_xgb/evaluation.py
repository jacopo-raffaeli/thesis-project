import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def metrics_cls_single(y_true, y_pred, y_proba) -> dict:
    return {
        "logloss": log_loss(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_proba, average="macro", multi_class="ovo"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def metrics_cls(y_true, y_pred_dict: dict, y_proba_dict: dict):
    metrics = {}
    for seed in y_pred_dict.keys():
        metrics[seed] = metrics_cls_single(y_true, y_pred_dict[seed], y_proba_dict[seed])

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "seed"
    df.columns.name = "metrics"

    return df


def mean_metrics_cls(df: pd.DataFrame):
    df = pd.concat(
        [
            df.mean().to_frame().T.rename(index={0: "mean"}),
            df.std().to_frame().T.rename(index={0: "std"}),
        ]
    )

    return df


def metrics_reg_single(y_true, y_pred) -> dict:
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def metrics_reg(y_true, y_pred_dict: dict):
    metrics = {}
    for seed in y_pred_dict.keys():
        metrics[seed] = metrics_reg_single(y_true, y_pred_dict[seed])

    df = pd.DataFrame.from_dict(metrics, orient="index")
    df.index.name = "seed"
    df.columns.name = "metrics"

    return df


def mean_model_metrics_reg(df: pd.DataFrame):
    df = pd.concat(
        [
            df.mean().to_frame().T.rename(index={0: "mean"}),
            df.std().to_frame().T.rename(index={0: "std"}),
        ]
    )

    return df


def model_metrics(y_true, y_pred_dict: dict, y_proba_dict: dict, exp: str) -> pd.DataFrame:
    match exp:
        case "reg":
            df = metrics_reg(y_true, y_pred_dict)
        case "cls":
            df = metrics_cls(y_true, y_pred_dict, y_proba_dict)
        case _:
            raise ValueError("Unknown importance analysis")

    return df


def mean_model_metrics(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    match exp:
        case "reg":
            df = mean_model_metrics_reg(df)
        case "cls":
            df = mean_metrics_cls(df)
        case _:
            raise ValueError("Unknown importance analysis")

    return df


def plot_cm(y_true, y_pred, labels: list[str], seed: int):
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".2f", cmap="Blues")

    ax.set_title(f"Confusion Matrix \nSeed: {seed}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    plt.show()


def plot_mean_cm(y_true, y_pred_dict, labels):
    cms = []

    for _, y_pred in y_pred_dict.items():
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize="true")
        cms.append(cm)

    mean_cm = np.mean(cms, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".2f", cmap="Blues", colorbar=True)

    ax.set_title("Mean Confusion Matrix Across Seeds")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    fig.tight_layout()
    plt.show()


def importance_xgb_ranking(importance_xgb, n_features, metric, seed):
    data = (
        importance_xgb.sort_values(by=metric, ascending=False)
        .reset_index(drop=True)
        .head(n_features)
    )
    data.index += 1
    data.index.name = "rank"
    cols = ["feature", metric]
    cols = cols + [c for c in data.columns if c not in cols]
    data = data[cols]
    print(f"Seed: {seed}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["feature"], data[metric])

    ax.invert_yaxis()
    ax.set_xlabel(metric.title())
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {n_features} features by {metric} \nSeed: {seed}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def importance_xgb_mean_ranking(importance_xgb_dict, n_features, metric):
    dfs = []

    for seed, df in importance_xgb_dict.items():
        tmp = df[["feature", metric]].copy()
        tmp["seed"] = seed
        dfs.append(tmp)

    all_importances = pd.concat(dfs, ignore_index=True)

    summary = (
        all_importances.groupby("feature")[metric]
        .agg(mean="mean", std="std")
        .fillna(0)
        .sort_values("mean", ascending=False)
        .head(n_features)
        .reset_index()
    )

    summary.index += 1
    summary.index.name = "rank"

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        summary["feature"],
        summary["mean"],
        xerr=summary["std"],
        capsize=3,
    )

    ax.invert_yaxis()
    ax.set_xlabel(f"Mean {metric.title()}")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {n_features} Features by Mean {metric.title()}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def selection_xgb_mean_ranking(selection_dict, n_features, metric):
    dfs = []

    for seed, df in selection_dict.items():
        tmp = df[["selected_feature", "iteration", metric]].copy()
        tmp["seed"] = seed
        dfs.append(tmp)

    all_selection = pd.concat(dfs, ignore_index=True)

    summary = (
        all_selection.groupby("selected_feature")
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            mean_it=("iteration", "mean"),
        )
        .fillna(0)
        .sort_values("mean", ascending=False)
        .head(n_features)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        summary["selected_feature"],
        summary["mean"],
        xerr=summary["std"],
        capsize=3,
    )

    ax.invert_yaxis()

    ax.set_xlabel(f"Mean {metric.title()}")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {n_features} Features by Mean Gain")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)

    fig.tight_layout()
    plt.show()


def selection_xgb_metrics(selection_hist_dict, metric):
    train_rmse_list = []
    val_rmse_list = []
    min_len_train = np.inf
    min_len_val = np.inf

    for seed, selection_hist in selection_hist_dict.items():
        train_rmse_list.append(selection_hist[f"train_{metric}"])
        val_rmse_list.append(selection_hist[f"val_{metric}"])
        min_len_train = min(min_len_train, len(selection_hist[f"train_{metric}"]))
        min_len_val = min(min_len_val, len(selection_hist[f"val_{metric}"]))

    for i, _ in enumerate(train_rmse_list):
        train_rmse_list[i] = train_rmse_list[i][:min_len_train]

    for i, _ in enumerate(val_rmse_list):
        val_rmse_list[i] = val_rmse_list[i][:min_len_val]

    train_rmse_mean = np.mean(train_rmse_list, axis=0)
    train_rmse_std = np.std(train_rmse_list, axis=0)
    val_rmse_mean = np.mean(val_rmse_list, axis=0)
    val_rmse_std = np.std(val_rmse_list, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    iterations = np.arange(min_len_train)
    ax.plot(iterations, train_rmse_mean, label=f"train {metric.upper()}")
    ax.fill_between(
        iterations,
        train_rmse_mean - train_rmse_std,
        train_rmse_mean + train_rmse_std,
        alpha=0.4,
    )

    iterations = np.arange(min_len_val)
    ax.plot(iterations, val_rmse_mean, label=f"validation {metric.upper()}")
    ax.fill_between(
        iterations,
        val_rmse_mean - 2 * val_rmse_std,
        val_rmse_mean + 2 * val_rmse_std,
        alpha=0.4,
    )

    ax.set_xlabel("Iteration number")

    ax.set_ylabel(f"{metric.upper()}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    ax.set_title(f"Iterative feature selection mean {metric.upper()}", size=12)

    fig.tight_layout()
    plt.show()


def feature_rankings(importance_xgb_dict, metric, show_rank=False):
    rankings = {}

    if show_rank:
        key = "rank"
    else:
        key = metric

    for seed, df in importance_xgb_dict.items():
        ranked = df.sort_values(metric, ascending=False).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        rankings[seed] = ranked.set_index("feature")[key]

    rankings = pd.DataFrame(rankings)
    rankings.columns.name = "seed"

    rankings["mean"] = rankings.mean(axis=1)
    rankings["std"] = rankings.std(axis=1)

    return rankings.sort_values("mean", ascending=show_rank)


def optuna_study_best_params(study_dict, model_dict):
    params_dict = {}
    for seed, study in study_dict.items():
        params = study.best_trial.params
        params["n_trees"] = model_dict[seed].get_booster().num_boosted_rounds()
        params_dict[seed] = params

    df = pd.DataFrame().from_dict(params_dict, orient="index")
    df.index.name = "seed"
    df.columns.name = "params"

    return df
