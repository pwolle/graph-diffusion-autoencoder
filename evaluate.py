from sample_symmetric import score_function, sample
from sigma_intevall import sigma_lower_bound, sigma_upper_bound
from plots import round_adj, plot
from models import BinaryEdgesModel
import functools
import jax
import jax.numpy as jnp
import jax.random as jrandom


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.npz")
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dim_at", type=int, default=8)
    parser.add_argument("--n_atoms", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--file_name", type=str, default="test")
    args = parser.parse_args()

    key = jrandom.PRNGKey(0)
    key, modul_key = jrandom.split(key)
    key, eval_key = jrandom.split(key)

    model = BinaryEdgesModel(modul_key, nlayer=2, dim=128, dim_at=8)

    evaluate(
        model,
        n_atoms=args.n_atoms,
        batch_size=args.batch_size,
        key=jrandom.PRNGKey(0),
        num_iterations=args.num_iterations,
        step_size=args.step_size,
        max_degree=args.max_degree,
        file_name=args.file_name,
    )
