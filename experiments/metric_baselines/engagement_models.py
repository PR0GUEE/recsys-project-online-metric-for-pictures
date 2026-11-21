"""Fit fixed pointwise, pairwise, and linear models on engagement targets."""

import numpy as np

from compare import CATBOOST_PARAMS


PAIRWISE_PARAMS = {**CATBOOST_PARAMS, "loss_function": "PairLogit"}
RIDGE_L2 = 0.01


def pointwise_predictions(x_train, target, x_val):
    """Fit the same CatBoost regressor used by the baseline comparison."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(x_train, target)
    return model.predict(x_val)


def pairwise_predictions(x_train, target, groups, x_val):
    """Let CatBoost generate preference pairs within contiguous requests."""
    from catboost import CatBoostRanker

    model = CatBoostRanker(**PAIRWISE_PARAMS)
    model.fit(x_train, target, group_id=groups)
    return model.predict(x_val)


def ridge_predictions(x_train, target, x_val):
    """Minimize mean squared loss plus L2 on coefficients of scaled features."""
    varying = np.ptp(x_train, axis=0) > 0
    train = x_train[:, varying].astype(float)
    feature_mean = train.mean(axis=0)
    feature_scale = train.std(axis=0)
    train = (train - feature_mean) / feature_scale
    val = (x_val[:, varying] - feature_mean) / feature_scale

    target_mean = target.mean()
    gram = train.T @ train / len(train)
    gram += RIDGE_L2 * np.eye(train.shape[1])
    covariance = train.T @ (target - target_mean) / len(train)
    coefficients = np.linalg.solve(gram, covariance)
    return target_mean + val @ coefficients
