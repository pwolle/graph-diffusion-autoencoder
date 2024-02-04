import datetime

import jax
import jax.numpy as jnp
import jax.random as jrandom
import memmpy

import wandb
from data import gdb13_graph_memmap
from models import BinaryEdgesModel
from plots import adj_to_graph, plotting, round_adj
from sample_symmetric import sample, score_function
from sigma_intevall import sigma_lower_bound, sigma_upper_bound


# @functools.partial(jax.jit, static_argnums=(0, 1, 2, 7))
def evaluate(
    samples,
    max_degree,
    file_name,
    use_wandb: bool = None,
):
    # Plot sample
    adjacency_matrices = round_adj(samples)
    graphs = adj_to_graph(adjacency_matrices)
    if use_wandb:
        return plotting(graphs, max_degree, file_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.npz")
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dim_at", type=int, default=8)
    parser.add_argument("--n_atoms", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--step_size", type=float, default=1e-4)
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--n_sigmas", type=int, default=15)
    parser.add_argument("--file_name", type=str, default="test")
    args = parser.parse_args()

    n_atoms = args.n_atoms
    seed = 0

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="sampling",
        config={
            "natoms": args.n_atoms,
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "step_size": args.step_size,
            "max_degree": args.max_degree,
            "n_sigmas": args.n_sigmas,
            "nlayer": args.nlayer,
            "dim": args.dim,
            "seed": seed,
            "start time": timestamp,
        },
    )

    data = gdb13_graph_memmap("data", args.n_atoms)
    data_valid = memmpy.split(data, "valid", shuffle=True, seed=seed)  # type: ignore

    key = jrandom.PRNGKey(seed)
    key, modul_key = jrandom.split(key)
    key, eval_key = jrandom.split(key)

    model = BinaryEdgesModel(modul_key, nlayer=2, dim=args.dim)
    model = model.load_leaves(args.model)

    score = score_function(model)
    score = jax.vmap(score)
    score = jax.jit(score)

    min_sigma = sigma_lower_bound(n_atoms)
    max_sigma = sigma_upper_bound(n_atoms)

    min_sigma = jnp.log(min_sigma)
    max_sigma = jnp.log(max_sigma)
    sigmas = jnp.logspace(
        min_sigma,
        max_sigma,
        args.n_sigmas,
        base=jnp.e,
    )[::-1]

    sample = jax.jit(sample, static_argnums=(1, 4, 5))

    samples = sample(
        sigmas=sigmas,
        score=score,
        key=key,
        n_atoms=n_atoms,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        step_size=args.step_size,
    )

    evaluate(
        samples=samples,
        max_degree=args.max_degree,
        file_name=args.file_name,
        use_wandb=True,
    )

    wandb.finish()
