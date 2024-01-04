import functools
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import lax
import matplotlib.pyplot as plt
from typeguard import typechecked
from typing import Callable, Tuple


def gradient_ascent(
    i: int,
    sample: Tuple[np.ndarray, float, float, np.ndarray],
    score: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Helper function for iteration of gradient ascent.

    Args:
        i (int): Iteration index.
        sample (Tuple[np.ndarray, float, float, np.ndarray]): (value, sigma, step, key)
            value (np.ndarray) - value of the sample
            sigma (float) - noise
            step (float) - step size
            key (np.ndarray) - random key
        score (Callable[[np.ndarray, float], np.ndarray]): Score function.

    Returns:
        Tuple[np.ndarray, float, float, np.ndarray]: (value, sigma, step, key)
            value (np.ndarray) - value of the sample
            sigma (float) - noise
            step (float) - step size
            key (np.ndarray) - random key
    """

    # Extract the values.
    value, sigma, step, key = sample
    key, subkey = random.split(key)

    # Generate the noise.
    noise = random.normal(key=subkey, shape=value.shape)

    # Update the samples.
    scr = score(value, sigma=sigma)
    value += step * scr + np.sqrt(step) * noise

    return value, sigma, step, key


def iterate_sigma(
    i: int,
    sample: Tuple[np.ndarray, np.ndarray],
    gradient_ascent: Callable[
        [int, Tuple[np.ndarray, float, float, np.ndarray]],
        Tuple[np.ndarray, float, float, np.ndarray],
    ],
    min_sigma: float,
    step_size: float,
    num_iterations: int,
    sigmas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function for iteration over different noise (sigma).

    Args:
        i (int): Iteration index.
        sample (Tuple[np.ndarray, np.ndarray]): (value, key)
            value (np.ndarray) - value of the sample
            key (np.ndarray) - random key
        gradient_ascent (Callable): Gradient ascent function.
        min_sigma (float): Minimum noise.
        step_size (float): Step size for Langevin dynamics.
        num_iterations (int): Number of iterations.
        sigmas (np.ndarray): List of noises.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (value, key)
            value (np.ndarray) - value of the sample
            key (np.ndarray) - random key
    """

    # Extract the values.
    value, key = sample
    sigma = sigmas[i]

    # Update the step size.
    step = step_size * sigma / min_sigma

    # Update the samples.
    value, *_ = lax.fori_loop(
        0,
        num_iterations,
        gradient_ascent,
        (value, sigma, step, key),
    )

    return value, key


@typechecked
def sample(
    sigmas: np.ndarray,
    score: Callable[[np.ndarray, float], np.ndarray],
    step_size: float,
    num_iterations: int = 1000,
    shape: Tuple[int, int] = (13, 13),
    key: np.ndarray = random.PRNGKey(0),
) -> np.ndarray:
    """
    Sample from the distribution using Langevin dynamics.

    Args:
        sigmas (np.ndarray): List of noises.
        score (Callable[[np.ndarray, float], np.ndarray]): Score function.
        step_size (float): Step size for Langevin dynamics.
        num_iterations (int): Number of iterations.
        shape (Tuple[int, int]): Shape of the samples.
        key (np.ndarray): Random key.

    Returns:
        np.ndarray: Samples.
    """
    # Initialize the samples.
    key, subkey = random.split(key)
    sample = random.normal(key=subkey, shape=shape)

    # Find the minimum standard deviation.
    min_std = sigmas[-1]

    # Initialize the interation functions.
    gradient_ascent_fixed = functools.partial(gradient_ascent, score=score)
    interate_sigma_fixed = functools.partial(
        iterate_sigma,
        gradient_ascent=gradient_ascent_fixed,
        min_sigma=min_std,
        step_size=step_size,
        num_iterations=num_iterations,
        sigmas=sigmas,
    )

    # Calculate the samples.
    sample, *_ = lax.fori_loop(
        0,
        len(sigmas),
        interate_sigma_fixed,
        (sample, key),
    )

    return sample
