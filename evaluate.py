from sample_symmetric import score_function, sample
from sigma_intevall import sigma_lower_bound, sigma_upper_bound
import functools
import jax
import jax.numpy as jnp


def evaluate(model, n_atoms, batch_size, key, num_iterations, step_size):
    score = score_function(model)
    score = jax.vmap(score)

    # sample = functools.partial(jax.jit(sample), static_argnums=(1, 4, 5))

    min_sigma = sigma_lower_bound(n_atoms)
    max_sigma = sigma_upper_bound(n_atoms)

    sigmas = jnp.linspace(min_sigma, max_sigma, num=10)

    # Sample
    samples = sample(
        sigmas=sigmas,
        score=score,
        key=key,
        n_atoms=n_atoms,
        batch_size=batch_size,
        num_iterations=num_iterations,
        step_size=step_size,
    )

    # Plot sample
    adjacency_matrices = samples.reshape(-1, n_atoms, n_atoms)
