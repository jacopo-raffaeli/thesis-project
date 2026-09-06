import numpy as np
import pandas as pd
import shap

from thesis_project.importance_xgb import registry, settings


def importance_shap(config_obj: settings.AnalysisConfig, model, data):
    explainer = shap.Explainer(model, data, seed=config_obj.seed)
    explanation = explainer(data)

    return explanation


def importance_xgb(
    model,
):
    booster = model.get_booster()
    feature_names = booster.feature_names

    importance_df = pd.DataFrame(index=feature_names)
    importance_df.index.name = "feature"

    for importance in registry.XGB_IMPORTANCES_REGISTRY:
        scores = booster.get_score(importance_type=importance)
        importance_df[importance] = pd.Series(scores).reindex(feature_names).fillna(np.nan)

    return importance_df
