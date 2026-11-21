"""Train click scores whose input features exclude the position in the log."""

import numpy as np

from compare import CATBOOST_PARAMS
from engagement_models import pointwise_predictions
from run_avatar_rerank import MAX_POSITION_WEIGHT


EM_STEPS = 3
PROBABILITY_EPSILON = 1e-6


def _examination_probability(click, exposure, relevance):
    """Posterior examination in the model C = E * R with independent E and R."""
    relevance = np.clip(relevance, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
    unclicked_posterior = exposure * (1 - relevance) / (1 - exposure * relevance)
    return np.where(click, 1.0, unclicked_posterior)


def _predict_exposure_click(x_train, click, exposure, x_predict):
    """Fit the position-based click model by three approximate EM steps.

    Start with a constant relevance matching mean click rate under the fixed
    exposure curve. Each M-step refits a finite boosted classifier with binary
    click labels and the examination posterior as the sample weight. Thus these
    are approximate M-steps. The returned score is P(R=1|q, post) under the
    click model; the exposure curve is only used during training.
    """
    from catboost import CatBoostClassifier

    relevance = click.mean() / exposure.mean()
    params = {**CATBOOST_PARAMS, "loss_function": "Logloss"}
    for step in range(EM_STEPS):
        sample_weight = _examination_probability(click, exposure, relevance)
        model = CatBoostClassifier(**params)
        model.fit(x_train, click, sample_weight=sample_weight)
        if step < EM_STEPS - 1:
            relevance = model.predict_proba(x_train)[:, 1]
    return model.predict_proba(x_predict)[:, 1]


def predict_click_models(x_train, frame, x_predict, click_curve) -> dict[str, np.ndarray]:
    """Compare raw labels, capped inverse-exposure labels and a click model.

    The caller supplies position-free feature matrices for both fitting and
    inference. The logged position in frame can affect the training signal.
    """
    click = frame["click"].to_numpy(dtype=bool)
    position = frame["position"].to_numpy(dtype=int) - 1
    exposure = np.asarray(click_curve["e_by_position"], dtype=float)[position]
    ips_target = click * np.minimum(MAX_POSITION_WEIGHT, 1 / exposure)
    return {
        "blind_raw_click": pointwise_predictions(x_train, click, x_predict),
        "blind_ips_click": pointwise_predictions(x_train, ips_target, x_predict),
        "blind_exposure_click": _predict_exposure_click(x_train, click, exposure, x_predict),
    }
