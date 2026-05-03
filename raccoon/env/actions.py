"""Action space utilities for OpenSpiel backgammon."""

import numpy as np

ACTION_SPACE_SIZE = 1352


def legal_action_mask(legal_actions: list[int]) -> np.ndarray:
    """Return a boolean mask of shape (1352,). True for legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[legal_actions] = True
    return mask
