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
    normal = jax.scipy.stats.norm.cdf(0.5)
    return 0.995 ** (2 / (natoms * (natoms - 1))) * 1 / normal


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
