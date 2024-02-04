import jax
import jax.numpy as jnp
import jax.random as jrandom
from typeguard import typechecked


@jax.jit
def sigma_lower_bound(natoms=10):
    """
    Lower bound for the noise of the Langevin dynamics.

    Parameters
    ---
    natoms: int
        Number of atoms.

    Returns
    ---
    float
        Lower bound for the noise of the Langevin dynamics.
    """
    quartile = 0.995 ** (2 / (natoms * (natoms - 1)))
    normal_quantile = jax.scipy.stats.norm.ppf(quartile)
    return 1 / (2 * normal_quantile)


@jax.jit
def sigma_upper_bound(natoms=10):
    """
    Upper bound for the noise of the Langevin dynamics.

    Parameters
    ---
    natoms: int
        Number of atoms.

    Returns
    ---
    float
        Upper bound for the noise of the Langevin dynamics.
    """
    return 0.707 * natoms
