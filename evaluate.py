from sample_symmetric import score_function, sample
from sigma_intevall import sigma_lower_bound, sigma_upper_bound
from plots import round_adj, plot
import functools
import jax
import jax.numpy as jnp


# @functools.partial(jax.jit, static_argnums=(0, 1, 2, 7))
def evaluate(
    model,
    n_atoms,
    batch_size,
    key,
    num_iterations,
    step_size,
    max_degree,
    file_name,
):
    score = score_function(model)
    score = jax.vmap(score)

    # sample = functools.partial(jax.jit(sample), static_argnums=(1, 4, 5))

    min_sigma = sigma_lower_bound(n_atoms)
    max_sigma = sigma_upper_bound(n_atoms)

    min_sigma = jnp.log(min_sigma)
    max_sigma = jnp.log(max_sigma)
    sigmas = jnp.logspace(min_sigma, max_sigma, 15, base=jnp.e)

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
    adjacency_matrices = round_adj(samples)
    plot(adjacency_matrices, max_degree, file_name)
