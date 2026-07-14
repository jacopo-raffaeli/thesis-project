import numpy as np
import pandas as pd
import shap

from thesis_project.importance_xgb import config


def importance_shap(config_obj: config.AnalysisConfig, model, data):
    explainer = shap.Explainer(model, data, seed=config_obj.seed)
    explanation = explainer(data)

    return explanation


def importance_xgb(
    model,
):
    importance_types = [
        "weight",  # the number of times a feature is used to split the data across all trees.
        "gain",  # the average gain across all splits the feature is used in.
        "cover",  # the average coverage across all splits the feature is used in.
        "total_gain",  # the total gain across all splits the feature is used in.
        "total_cover",  # the total coverage across all splits the feature is used in.
    ]

    booster = model.get_booster()
    feature_names = booster.feature_names

    importance_df = pd.DataFrame(index=feature_names)
    importance_df.index.name = "feature"

    for importance in importance_types:
        scores = booster.get_score(importance_type=importance)
        importance_df[importance] = pd.Series(scores).reindex(feature_names).fillna(np.nan)

    importance_df["used"] = importance_df["weight"] != np.nan
    importance_df = importance_df.reset_index()

    return importance_df
