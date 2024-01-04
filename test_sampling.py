import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import jax.scipy.stats as stats
import matplotlib.pyplot as plt

from langevin_dynamic import sample


def gaussian_mixture(
    x: np.ndarray,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = ([2, 2]),
    var_1: float = 2,
    perturb_var: float = 0.1,
    key=random.PRNGKey(42),
):
    cov_0 = var_0 * np.eye(2)
    multi_0 = stats.multivariate_normal.pdf(
        x,
        mean=mean_0,
        cov=cov_0,
    )

    cov_1 = var_1 * np.eye(2)
    multi_1 = stats.multivariate_normal.pdf(
        x,
        mean=mean_1,
        cov=cov_1,
    )
    return 0.2 * multi_0 + 0.8 * multi_1


def log_gaussian_mixture(
    x: np.ndarray,
    mean_0: np.ndarray = np.array([-3, -3], dtype=np.float32),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2], dtype=np.float32),
    var_1: float = 2,
):
    mixture = gaussian_mixture(x, mean_0, var_0, mean_1, var_1)
    return np.log(mixture)


def score_gaussian_mixture(x):
    return grad(log_gaussian_mixture)(x)


def plot_gaussian_mixture(
    mean_0,
    var_0,
    mean_1,
    var_1,
    perturb_var,
    key,
    num_points=40,
    ax=plt,
):
    x1 = np.linspace(-5, 5, num_points)
    x2 = np.linspace(-5, 5, num_points)

    X, Y = np.array(np.meshgrid(x1, x2))
    x = np.dstack((X, Y))

    key, subkey = random.split(key)
    y = np.array(
        [
            gaussian_mixture(i, mean_0, var_0, mean_1, var_1, perturb_var, subkey)
            for i in x.reshape(num_points**2, 2)
        ]
    )
    y = y.reshape(num_points, num_points)

    ax.contourf(x1, x2, y, alpha=0.7)


# def p_0(x):
#     n_1 = normal(mu=(0, 0), sigma=0.1)
#     n_2 = normal(mu=(1, 1), sigma=0.1)
#     return 0.1 * n_1(x) + 0.9 * n_2(x)


# def p(x, sigma):
#     n_1_sigma = (0.1**2 + sigma**2) ** 0.5

#     n_1 = normal(mu=(0, 0), sigma=n_sigma)
#     n_2 = normal(mu=(1, 1), sigma=n_sigma)
#     return 0.1 * n_1(x) + 0.9 * n_2(x)


def gaussian_mixture_noise(
    x: np.ndarray,
    sigma: float,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = ([2, 2]),
    var_1: float = 2,
    key=random.PRNGKey(42),
):
    sigma_0 = var_0 + sigma**2
    cov_0 = sigma_0 * np.eye(2)
    multi_0 = stats.multivariate_normal.pdf(
        x,
        mean=mean_0,
        cov=cov_0,
    )

    sigma_1 = var_1 + sigma**2
    cov_1 = sigma_1 * np.eye(2)
    multi_1 = stats.multivariate_normal.pdf(
        x,
        mean=mean_1,
        cov=cov_1,
    )
    return 0.2 * multi_0 + 0.8 * multi_1


def log_gaussian_mixture_noise(
    x: np.ndarray,
    sigma: float,
    mean_0: np.ndarray = np.array([-3, -3]),
    var_0: float = 1,
    mean_1: np.ndarray = np.array([2, 2]),
    var_1: float = 2,
):
    mixture = gaussian_mixture_noise(x, sigma, mean_0, var_0, mean_1, var_1)
    return np.log(mixture)


def perturb(x, key, var=0.1):
    return x + random.normal(key, shape=x.shape) * var


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
