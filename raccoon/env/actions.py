"""Action space utilities for OpenSpiel backgammon."""

import numpy as np

from raccoon.env.game_wrapper import GameState

ACTION_SPACE_SIZE = 1352


def legal_action_mask(legal_actions: list[int]) -> np.ndarray:
    """Return a boolean mask of shape (1352,). True for legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[legal_actions] = True
    return mask


def action_to_string(state: GameState, action: int) -> str:
    """Human-readable move description (e.g., '24/23 13/11')."""
    return state.action_to_string(action)
