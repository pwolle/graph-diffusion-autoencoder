import functools
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import lax
import matplotlib.pyplot as plt
from typeguard import typechecked


def gradient_ascent(i, sample, score):
    """
    Helper function for iteration of gradient ascent.

    Args:
        i (int): Iteration index.
        sample (np.ndarray): (key, value, step).
        score (callable): Score function.
    """
    value, sigma, step, key = sample
    key, subkey = random.split(key)
    noise = random.normal(key=subkey, shape=value.shape)
    scr = score(value, sigma=sigma)
    value += step * scr + np.sqrt(step) * noise
    return value, sigma, step, key


def iterate_sigma(
    i,
    sample,
    gradient_ascent,
    min_sigma,
    step_size,
    num_iterations,
    sigmas,
):
    """
    Helper function for iteration of gradient ascent.

    Args:
        i (int): Iteration index.
        sample (np.ndarray): (key, value, step).
        score (callable): Score function.
    """
    value, key = sample
    sigma = sigmas[i]

    step = step_size * sigma / min_sigma

    value, *_ = lax.fori_loop(
        0,
        num_iterations,
        gradient_ascent,
        (value, sigma, step, key),
    )

    return value, sigmas, key


@typechecked
def sample(
    sigmas: np.ndarray,
    score: callable,
    step_size: float,
    num_iterations: int = 1000,
    shape: tuple[int] = (13, 13),
    key: np.ndarray = random.PRNGKey(0),
):
    """Langevin dynamics for sampling from a Gaussian distribution.

    Args:
        std_deviations (list[int]): Standard deviations of the Gaussian distribution.
        score_functions (list[callable]): Score functions of the Gaussian distribution.
        step_size (float): Step size for Langevin dynamics.
        num_iterations (int, optional): Number of iterations. Defaults to 1000.
        shape (tuple[int], optional): Shape of the samples. Defaults to (N, N).
        key (np.ndarray, optional): Random key for generating noise. Defaults to random.PRNGKey(0).

    Returns:
        np.ndarray: Samples from the Gaussian distribution.
    """
    # Initialize the samples.
    key, subkey = random.split(key)
    sample = random.normal(key=subkey, shape=shape)

    min_std = sigmas[-1]

    gradient_ascent_fixed = functools.partial(gradient_ascent, score=score)

    interate_sigma_fixed = functools.partial(
        iterate_sigma,
        gradient_ascent=gradient_ascent_fixed,
        min_sigma=min_std,
        step_size=step_size,
        num_iterations=num_iterations,
        sigmas=sigmas,
    )

    # Iterate over standard deviations.

    sample, *_ = lax.fori_loop(
        0,
        len(sigmas),
        interate_sigma_fixed,
        (sample, key),
    )

    return sample


# for _ in range(num_iterations):
#             key, subkey = random.split(key)
#             noise = random.normal(key=subkey, shape=shape)
#             scr = score(sample, sigma)
#             sample += step * scr + np.sqrt(step) * noise


# for sigma in sigmas:
#         # Update the samples.
#         step = step_size * sigma / min_std

#         # Iterate over iterations.
#         sample, *_ = lax.fori_loop(
#             0,
#             num_iterations,
#             gradient_ascent_fixed,
#             (sample, sigma, step, key),
#         )
