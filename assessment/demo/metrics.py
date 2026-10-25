import os

"""One-to-one matching between estimated audio beats and detected visual events."""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Compare detected visual events with audio beat times


def score(audio_events, visual_events, tolerance=0.125):
    audio = np.asarray(audio_events)
    visual = np.asarray(visual_events)
    count = 0
    if len(audio) and len(visual):
        distance = np.abs(audio[:, None] - visual[None, :])
        penalty = (min(len(audio), len(visual)) + 1) * (tolerance + 1)
        row, col = linear_sum_assignment(
            np.where(distance <= tolerance, distance, penalty)
        )
        count = int((distance[row, col] <= tolerance).sum())
    return {
        "precision": count / len(visual) if len(visual) else 0.0,
        "coverage": count / len(audio) if len(audio) else 0.0,
        "f1": (
            2 * count / (len(audio) + len(visual)) if len(audio) + len(visual) else 0.0
        ),
        "matches": count,
    }
