import jax
import jax.numpy as np
from jax import grad
import jax.random as jrandom
import jax.scipy.stats as stats
import matplotlib.pyplot as plt

from sample import sample
from typing import Tuple


def gaussian_mixture_density(
    x: np.ndarray,
    noise_scale: float,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = ([2, 2]),
    var_1: float = 2,
) -> np.ndarray:
    """
    Compute the probability density function of a Gaussian mixture with noise.

    Parameters
    ---
    x: (np.ndarray)
        Input array of shape (n, 2).

    noise_scale: float
        Standard deviation of the noise.

    mean_0: np.ndarray, optional, default: np.array([-3, -3])
        Mean of the first Gaussian

    var_0: float, optional, default: 1
        Variance of the first Gaussian

    mean_1: np.ndarray, optional, default: np.array([2, 2])
        Mean of the second Gaussian

    var_1: float, optional, default: 2
        Variance of the second Gaussian

    Returns
    ---
    np.ndarray
        Array of shape (n,)
        representing the probability density function of the Gaussian mixture noise.
    """
    # compute first gaussian
    sigma_0 = var_0 + noise_scale**2
    cov_0 = sigma_0 * np.eye(2)
    multi_0 = stats.multivariate_normal.pdf(
        x,
        mean=mean_0,
        cov=cov_0,
    )

    # compute second gaussian
    sigma_1 = var_1 + noise_scale**2
    cov_1 = sigma_1 * np.eye(2)
    multi_1 = stats.multivariate_normal.pdf(
        x,
        mean=mean_1,
        cov=cov_1,
    )

    # return the mixture
    return 0.2 * multi_0 + 0.8 * multi_1


def log_gaussian_mixture(
    x: np.ndarray,
    sigma: float,
    mean_0: np.ndarray = np.array([-3, -3], dtype=np.float32),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2], dtype=np.float32),
    var_1: float = 2,
) -> np.ndarray:
    """
    Compute the log of a Gaussian mixture noise.

    Parameters
    ---
    x: np.ndarray
        Input array of shape (n, 2).

    sigma: float
        Standard deviation of the noise.

    mean_0: np.ndarray, optional, default: np.array([-3, -3])
        Mean of the first Gaussian

    var_0: float, optional, default: 1
        Variance of the first Gaussian

    mean_1: np.ndarray, optional, default: np.array([2, 2])
        Mean of the second Gaussian

    var_1: float, optional, default: 2
        Variance of the second Gaussian

    Returns
    ---
    np.ndarray
        Array of shape (n,)
        representing the log of the probability density function of
        the Gaussian mixture noise.
    """
    # compute the mixture
    mixture = gaussian_mixture_density(x, sigma, mean_0, var_0, mean_1, var_1)

    # return the log of the mixture
    return np.log(mixture)


def plot_gaussian_mixture(
    perturb_var: float = 0,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2]),
    var_1: float = 2,
    num_points: int = 40,
    ax: plt.Axes = plt,
) -> None:
    """
    Plots a 2D Gaussian mixture.

    Parameters
    ---
    perturb_var: float, optional, default: 0
        Variance of the noise.

    mean_0: np.ndarray, optional, default: np.array([-3, -3])
        Mean of the first Gaussian

    var_0: float, optional, default: 1
        Variance of the first Gaussian

    mean_1: np.ndarray, optional, default: np.array([2, 2])
        Mean of the second Gaussian

    var_1: float, optional, default: 2
        Variance of the second Gaussian

    num_points: int, optional, default: 40
        Number of points to plot.

    ax: plt.Axes, optional, default: plt
        Matplotlib axes to plot on.

    Returns
    ---
    None
    """
    # Create the grid.
    x1 = np.linspace(-5, 5, num_points)
    x2 = np.linspace(-5, 5, num_points)
    X, Y = np.array(np.meshgrid(x1, x2))
    x = np.dstack((X, Y))

    # Compute the mixture.
    y = np.array(
        jax.vmap(
            lambda i: gaussian_mixture_density(
                i,
                perturb_var,
                mean_0,
                var_0,
                mean_1,
                var_1,
            ),
            in_axes=(0),
            out_axes=0,
        )(x.reshape(num_points**2, 2))
    )
    y = y.reshape(num_points, num_points)

    # Plot the mixture.
    ax.contourf(x1, x2, y, alpha=0.7)


if __name__ == "__main__":
    key = jrandom.PRNGKey(42)

    plot_gaussian_mixture(
        mean_0=np.array([-3, -3]),
        var_0=1,
        mean_1=np.array([2, 2]),
        var_1=2,
        perturb_var=0,
        num_points=10,
    )

    score = grad(log_gaussian_mixture)
    samples = []

    for _ in range(150):
        key, subkey = jrandom.split(key)
        smpl = sample(
            sigmas=np.array([4, 2, 0.5, 0.2, 0.01]),
            score=score,
            step_size=1e-4,
            num_iterations=30,
            shape=(2,),
            key=subkey,
        )
        samples.append(smpl)

        plt.scatter(
            smpl[0],
            smpl[1],
            c="red",
            s=10,
            alpha=1,
            label="sample point",
        )

    # plot sample point
    plt.show()
