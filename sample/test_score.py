import jax.numpy as np
import jax.scipy.stats as stats
import jax.random as jrandom
import matplotlib.pyplot as plt

from langevin_dynamics import sample
from langevin_dynamics import score_function


def test_2d_p(x, sigma):
    p_0 = test_probability(x[0])
    p_1 = test_probability(x[1])
    return np.array([p_0, p_1])


def test_probability(x, sigma):
    """
    Compute the probability density function of a Gaussian mixture with noise.

    Parameters
    ---
    x: (np.ndarray)
        Input array of shape (n, 2).

    sigma: float
        Standard deviation of the noise.

    Returns
    ---
    np.ndarray
        Array of shape (n,)
        representing the probability density function of the Gaussian mixture noise.
    """
    prob_0 = 1 / 3 * stats.norm.pdf(x, loc=0, scale=sigma)
    prob_1 = 2 / 3 * stats.norm.pdf(x, loc=1, scale=sigma)
    return prob_1 / (prob_0 + prob_1)


if __name__ == "__main__":
    score = score_function(test_probability)

    key = jrandom.PRNGKey(0)

    for _ in range(20):
        subkey, key = jrandom.split(key)
        samples = sample(
            sigmas=np.array([4, 2, 1, 0.5, 0.2, 0.1]),
            score=score,
            step_size=1e-3,
            num_iterations=50,
            shape=(2,),
            key=subkey,
        )
        print(samples)

    # Plot histogram
    # plt.hist(samples, bins=100)
    # plt.show()
