"""Fit rank-independent scores from incomplete engagement observations."""

import numpy as np

from compare import _request_starts
from run_avatar_rerank import MAX_POSITION_WEIGHT
from run_engagement import COMPOSITES, composite, signals
from run_reward_ablation import ETA, LOCAL_PAIR_STRENGTH, fit_pairwise


def _preference_pairs(frame, click_curve):
    logged = signals(frame)
    reward = composite(logged, COMPOSITES["strong_actions"])
    clicked = logged["click"].astype(bool)
    liked = logged["like"].astype(bool)
    inactive = ~(clicked | logged["active"].astype(bool))
    strong = liked | logged["collect"].astype(bool) | logged["long_view"].astype(bool)
    position = frame.position.to_numpy(dtype=int)
    exposure = np.asarray(click_curve["e_by_position"], dtype=float)[position - 1]
    mild = 1 + ETA * (np.minimum(MAX_POSITION_WEIGHT, 1 / exposure) - 1)

    pair_blocks, control_blocks, augmented_blocks = [], [], []
    local_pairs = requests_with_local_pairs = local_unclicked_like_pairs = 0
    added_pair_weight = 0.0
    cuts = np.append(_request_starts(frame), len(frame))
    for start, end in zip(cuts[:-1], cuts[1:]):
        engaged_rows = start + np.flatnonzero(~inactive[start:end])
        inactive_rows = start + np.flatnonzero(inactive[start:end])
        if not len(engaged_rows) or not len(inactive_rows):
            continue

        winner = np.repeat(engaged_rows, len(inactive_rows))
        loser = np.tile(inactive_rows, len(engaged_rows))
        control = reward[winner]
        control /= control.sum()
        augmented = control.copy()
        gap = position[winner] - position[loser]
        local = strong[winner] & (gap >= 1) & (gap <= 2)
        count = int(local.sum())
        if count:
            bonus = LOCAL_PAIR_STRENGTH * exposure[loser[local]] * mild[winner[local]] / count
            augmented[local] += bonus
            augmented /= augmented.sum()
            local_pairs += count
            local_unclicked_like_pairs += int((local & liked[winner] & ~clicked[winner]).sum())
            requests_with_local_pairs += 1
            added_pair_weight += float(bonus.sum())

        pair_blocks.append(np.column_stack((winner, loser)).astype(np.int32))
        control_blocks.append(control)
        augmented_blocks.append(augmented)

    pairs = np.concatenate(pair_blocks) if pair_blocks else np.empty((0, 2), dtype=np.int32)
    control = np.concatenate(control_blocks) if control_blocks else np.empty(0)
    augmented = np.concatenate(augmented_blocks) if augmented_blocks else np.empty(0)
    coverage = {
        "preference_pairs": len(pairs),
        "requests_with_preference_pairs": len(pair_blocks),
        "local_pairs": local_pairs,
        "requests_with_local_pairs": requests_with_local_pairs,
        "local_unclicked_like_pairs": local_unclicked_like_pairs,
        "extra_pair_weight_before_normalization": added_pair_weight,
    }
    return reward, pairs, control, augmented, coverage


def predict_preference_models(x_train, frame, x_predict, click_curve):
    """Fit two PairLogit arms on position-free inputs supplied by the caller.

    Each request with pairs has unit total weight. Both arms rank every engaged
    result over inactive results. The control weights wins by raw engagement.
    The augmented arm adds weight to an inactive result one or two original
    places above a strong engagement, then normalizes within that request.
    """
    reward, pairs, control, augmented, coverage = _preference_pairs(frame, click_curve)
    groups = frame.request_key.to_numpy()
    predictions = {}
    for name, weights in (("blind_pair_control", control), ("blind_pair_skip", augmented)):
        if len(pairs):
            predictions[name] = fit_pairwise(x_train, reward, groups, pairs, weights, x_predict)
        else:
            predictions[name] = np.zeros(len(x_predict))
    return predictions, coverage
