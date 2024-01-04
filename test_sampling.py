import jax.numpy as np
from jax import grad
from jax import random
import jax.scipy.stats as stats
import matplotlib.pyplot as plt

from langevin_dynamic import sample
from typing import Tuple


def gaussian_mixture_noise(
    x: np.ndarray,
    sigma: float,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = ([2, 2]),
    var_1: float = 2,
) -> np.ndarray:
    """
    Compute the probability density function of a Gaussian mixture with noise.

    Parameters:
        x (np.ndarray): Input array of shape (n, 2).
        sigma (float): Standard deviation of the noise.
        mean_0 (np.ndarray, optional): Mean of the first Gaussian. Defaults to np.array([-3, -3]).
        var_0 (float, optional): Variance of the first Gaussian. Defaults to 1.
        mean_1 (np.ndarray, optional): Mean of the second Gaussian. Defaults to np.array([2, 2]).
        var_1 (float, optional): Variance of the second Gaussian. Defaults to 2.

    Returns:
        np.ndarray: Array of shape (n,) representing the probability density function of the Gaussian mixture noise.
    """
    # compute first gaussian
    sigma_0 = var_0 + sigma**2
    cov_0 = sigma_0 * np.eye(2)
    multi_0 = stats.multivariate_normal.pdf(
        x,
        mean=mean_0,
        cov=cov_0,
    )

    # compute second gaussian
    sigma_1 = var_1 + sigma**2
    cov_1 = sigma_1 * np.eye(2)
    multi_1 = stats.multivariate_normal.pdf(
        x,
        mean=mean_1,
        cov=cov_1,
    )

    # return the mixture
    return 0.2 * multi_0 + 0.8 * multi_1


def log_gaussian_mixture_noise(
    x: np.ndarray,
    sigma: float,
    mean_0: np.ndarray = np.array([-3, -3], dtype=np.float32),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2], dtype=np.float32),
    var_1: float = 2,
) -> np.ndarray:
    """
    Compute the log of a Gaussian mixture noise.

    Parameters:
    - x: Input array.
    - sigma: Standard deviation of the noise.
    - mean_0: Mean of the first Gaussian component.
    - var_0: Variance of the first Gaussian component.
    - mean_1: Mean of the second Gaussian component.
    - var_1: Variance of the second Gaussian component.

    Returns:
    - np.ndarray: Array containing the log of the Gaussian mixture noise.
    """
    # compute the mixture
    mixture = gaussian_mixture_noise(x, sigma, mean_0, var_0, mean_1, var_1)

    # return the log of the mixture
    return np.log(mixture)


def plot_gaussian_mixture(
    perturb_var: float = 0,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2]),
    var_1: float = 2,
    key: random.PRNGKey = random.PRNGKey(42),
    num_points: int = 40,
    ax: plt.Axes = plt,
) -> None:
    """
    Plots a 2D Gaussian mixture.

    Parameters:
        perturb_var (float): Variance of the perturbation noise.
        mean_0 (np.ndarray): Mean of the first Gaussian component.
        var_0 (float): Variance of the first Gaussian component.
        mean_1 (np.ndarray): Mean of the second Gaussian component.
        var_1 (float): Variance of the second Gaussian component.
        key (random.PRNGKey): Random key for generating random numbers.
        num_points (int): Number of points in the grid.
        ax (plt.Axes): Axes object to plot the mixture on.

    Returns:
        None
    """
    # Create the grid.
    x1 = np.linspace(-5, 5, num_points)
    x2 = np.linspace(-5, 5, num_points)
    X, Y = np.array(np.meshgrid(x1, x2))
    x = np.dstack((X, Y))

    # Compute the mixture.
    key, subkey = random.split(key)
    y = np.array(
        [
            gaussian_mixture_noise(
                i,
                perturb_var,
                mean_0,
                var_0,
                mean_1,
                var_1,
                subkey,
            )
            for i in x.reshape(num_points**2, 2)
        ]
    )
    y = y.reshape(num_points, num_points)

    # Plot the mixture.
    ax.contourf(x1, x2, y, alpha=0.7)


if __name__ == "__main__":
    key = random.PRNGKey(42)
    key, subkey = random.split(key)

    plot_gaussian_mixture(
        mean_0=np.array([-3, -3]),
        var_0=1,
        mean_1=np.array([2, 2]),
        var_1=2,
        perturb_var=0,
        key=subkey,
        num_points=10,
    )

    # plt.show()

    # fig, axs = plt.subplots(2, 2)

    # plot_gaussian_mixture(
    #     mean_0=np.array([-3, -3]),
    #     var_0=1,
    #     mean_1=np.array([2, 2]),
    #     var_1=2,
    #     perturb_var=0,
    #     key=subkey,
    #     ax=axs[0, 0],
    # )
    # axs[0, 0].set_title("Perturb Var = 0")

    # key, subkey = random.split(key)
    # plot_gaussian_mixture(
    #     mean_0=np.array([-3, -3]),
    #     var_0=1,
    #     mean_1=np.array([2, 2]),
    #     var_1=2,
    #     perturb_var=100,
    #     key=subkey,
    #     ax=axs[0, 1],
    # )
    # axs[0, 1].set_title("Perturb Var = 1")

    # key, subkey = random.split(key)
    # plot_gaussian_mixture(
    #     mean_0=np.array([-3, -3]),
    #     var_0=1,
    #     mean_1=np.array([2, 2]),
    #     var_1=2,
    #     perturb_var=2,
    #     key=subkey,
    #     ax=axs[1, 0],
    # )
    # axs[1, 0].set_title("Perturb Var = 2")

    # key, subkey = random.split(key)
    # plot_gaussian_mixture(
    #     mean_0=np.array([-3, -3]),
    #     var_0=1,
    #     mean_1=np.array([2, 2]),
    #     var_1=2,
    #     perturb_var=5,
    #     key=subkey,
    #     ax=axs[1, 1],
    # )
    # axs[1, 1].set_title("Perturb Var = 3")

    # plt.tight_layout()

    # plt.show()
    # print(log_gaussian_mixture(np.array([1, 1])))
    score = grad(log_gaussian_mixture_noise)

    # key = random.PRNGKey(42)
    key, subkey = random.split(key)

    samples = []

    for _ in range(50):
        key, subkey = random.split(key)
        smpl = sample(
            sigmas=np.array([1, 0.5, 0.1]),
            score=score,
            step_size=0.1,
            num_iterations=20,
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
        # print(smpl)

    # print(samples)

    # plot sample point
    plt.show()
