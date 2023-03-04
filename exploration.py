"""
This file includes several exploration methods
such as epsilon_greedy and boltzmann_greedy.
"""

import random
import numpy as np
import numpy.typing as npt


def epsilon_greedy(actions: tuple,
                   weights: npt.NDArray[np.float32],
                   epsilon: float):
    """
    Epsilon_greedy exploration method.

    :arg actions:
        tuple of possible actions.

    :arg weights:
        numpy array(float) of weights for each possible action

    :arg epsilon:
        constant value of epsilon

    :returns:
        chosen action among actions
    """
    rnd_value = random.random()
    if rnd_value < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(weights)]


def boltzmann(actions: tuple, weights: npt.NDArray[np.float32], tau: float):
    """
    Boltzmann greedy exploration method.

    :arg actions:
        tuple of possible actions.

    :arg weights:
        numpy array(float) of weights for each possible action

    :arg tau:
        constant value of tau

    :returns:
        chosen action among actions
    """
    max_weight = np.max(weights)
    exp_weights = np.exp((weights - max_weight) / tau)
    sum_exp_weights = np.sum(exp_weights)
    final_weights = exp_weights / sum_exp_weights
    return random.choices(actions, weights=final_weights, k=1)[0]
