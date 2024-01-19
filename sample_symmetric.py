import functools
import jax.numpy as np
import jax
import jax.random as jrandom
import jax.nn as jnn
from jax import lax
import typeguard
from typeguard import typechecked
from typing import Callable

from models import symmetric_normal, set_diagonal


# @functools.partial(jax.jit, static_argnums=(2))
def langevin_dynamics_step(
    i: int,
    sample: tuple[np.ndarray, float, float, np.ndarray],
    score: Callable[[np.ndarray, float], np.ndarray],
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    Helper function for a Langevin dynamics step.

    Parameters
    ---
    i: int
        Iteration index.

    sample: tuple[np.ndarray, float, float, np.ndarray]
        (value, sigma, step, key)
            value (np.ndarray) - value of the sample
            sigma (float) - noise
            step (float) - step size
            key (np.ndarray) - random key

    score: Callable[[np.ndarray, float], np.ndarray]
        Score function of distribution.

    Returns
    ---
    tuple[np.ndarray, float, float, np.ndarray]
        (value, sigma, step, key)
            value (np.ndarray) - value of the sample
            sigma (float) - noise
            step (float) - step size
            key (np.ndarray) - random key
    """

    # Extract the values.
    value, sigma, step, key = sample
    key, subkey = jrandom.split(key)

    # Generate the noise.
    noise = symmetric_normal(key=subkey, shape=value.shape)

    # Update the samples.
    score_value = score(value, sigma=sigma)
    value += step * score_value + np.sqrt(step) * noise

    return value, sigma, step, key


# @functools.partial(jax.jit, static_argnums=(2, 7))
def iterate_for_fixed_sigma(
    i: int,
    sample: tuple[np.ndarray, np.ndarray],
    langevin_dynamics_step: Callable[
        [int, tuple[np.ndarray, float, float, np.ndarray]],
        tuple[np.ndarray, float, float, np.ndarray],
    ],
    min_sigma: float,
    step_size: float,
    num_iterations: int,
    sigmas: np.ndarray,
    batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function for iteration over different noise (sigma).

    Parameters
    ---
    i: int
        Iteration index.

    sample: tuple[np.ndarray, np.ndarray]
        (value, key)
            value (np.ndarray) - value of the sample
            key (np.ndarray) - random key

    langevin_dynamics_step: Callable
        Function for a Lagevin dynamics step.

    min_sigma: float
        Smallest noise scale.

    step_size: float
        smallest step size for Langevin dynamics.

    num_iterations: int
        Number of iterations of Lagevin dynamics steps per noise scale.

    sigmas: np.ndarray
        List of noises scales.

    Returns
    ---
    tuple[np.ndarray, np.ndarray]
        (value, key)
            value (np.ndarray) - value of the sample
            key (np.ndarray) - random key
    """

    # Extract the values.
    value, key = sample
    sigma = sigmas[i]

    # Update the step size.
    step = step_size * sigma**2 / min_sigma**2

    sigma_batch = np.ones((batch_size,)) * sigma

    # Update the samples.
    value, *_ = lax.fori_loop(
        0,
        num_iterations,
        langevin_dynamics_step,
        (value, sigma_batch, step, key),
    )

    return value, key


@typechecked
# @functools.partial(jax.jit, static_argnums=(1, 4, 5))
def sample(
    sigmas: np.ndarray,
    score: Callable[[np.ndarray, float], np.ndarray],
    step_size: float = 0.01,
    num_iterations: int = 1000,
    n_atoms: int = 10,
    batch_size: int = 1,
    key: jrandom.PRNGKey = jrandom.PRNGKey(0),
) -> np.ndarray:
    """
    Sample from the distribution using Langevin dynamics.

    Is jittable with static arguments: 1, 4, 5

    Parameters
    ---
    sigmas: np.ndarray
        List of noise scales.

    score: Callable[[np.ndarray, float], np.ndarray]
        Score function of distribution.

    step_size: float
        smallest step size for Langevin dynamics.

    num_iterations: int
        Number of iterations.

    shape: tuple[int, int]
        Shape of the samples.

    key: np.ndarray
        Random key.

    Returns
    ---
    np.ndarray
        Sample.
    """
    # Initialize the samples.
    key, subkey = jrandom.split(key)
    sample = symmetric_normal(key=subkey, shape=(batch_size, n_atoms, n_atoms))

    # Find the minimum standard deviation.
    min_std = sigmas[-1]

    # Initialize the interation functions.
    langevin_dynamics_fixed = functools.partial(
        langevin_dynamics_step,
        score=score,
    )
    interate_for_fixed_sigma_fixed = functools.partial(
        iterate_for_fixed_sigma,
        langevin_dynamics_step=langevin_dynamics_fixed,
        min_sigma=min_std,
        step_size=step_size,
        num_iterations=num_iterations,
        sigmas=sigmas,
        batch_size=batch_size,
    )

    # Calculate the samples.
    sample, *_ = lax.fori_loop(
        0,
        len(sigmas),
        interate_for_fixed_sigma_fixed,
        (sample, key),
    )
    # Set diagonal to zero.

    sample = jax.vmap(set_diagonal)(sample, np.zeros((batch_size,)))
    return sample


def score_function(probability):
    """
    Compute the score function for x + z.
    With z ~ N(0, sigma^2).and x = 0 or 1.
    Given the probability density function of x = 1 given x + z.

    Parameters
    ---
    probability: Callable[[np.array], np.array]
        Probability density function.

    Returns
    ---
    Callable[[np.array, np.array], np.array]
        Score function.
    """

    def score(x_noise, sigma):
        prob = probability(x_noise, sigma)
        prob = jnn.sigmoid(prob)
        return prob / sigma**2 - x_noise / sigma**2

    return score
