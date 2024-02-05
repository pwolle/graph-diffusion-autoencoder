from sample_symmetric import sample, score_function
from models import BinaryEdgesModel, GraphDiffusionAutoencoder
from sigma_intevall import sigma_lower_bound, sigma_upper_bound
from evaluate import evaluate
from data import gdb13_graph_memmap
from sample_symmetric import to_conditianal_probability

import jax
import jax.random as jrandom
import jax.numpy as jnp
import wandb
import datetime
import memmpy
import networkx as nx
import matplotlib.pyplot as plt
import functools

if __name__ == "__main__":
    n_atoms = 12
    seed = 0
    batch_size = 16
    max_degree = 3
    dim = 256
    nlayer = 2

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="sampling",
        config={
            "natoms": n_atoms,
            "batch_size": batch_size,
            "max_degree": max_degree,
            "nlayer": nlayer,
            "dim": dim,
            "seed": seed,
            "start time": timestamp,
        },
    )

    # Adjacency matrix
    data = gdb13_graph_memmap("data", n_atoms)

    data_train = memmpy.split(data, "train", shuffle=True, seed=seed)  # type: ignore
    data_train = memmpy.Batched(data_train, batch_size, True)

    data_valid = memmpy.split(data, "valid", shuffle=True, seed=seed)  # type: ignore
    data_valid = memmpy.unwrap(data_valid)[: 1024 * 4]

    adjacency = jnp.array(data_valid[0])
    graph = nx.from_numpy_array(adjacency)
    nx.draw(graph)
    graph_image = wandb.Image(plt)

    wandb.log({"graph": graph_image})

    # unconditional model
    key = jrandom.PRNGKey(seed)
    key, modul_unc_key = jrandom.split(key)

    model_unconditional = BinaryEdgesModel(modul_unc_key, nlayer=nlayer, dim=dim)
    model_unconditional = model_unconditional.load_leaves("model10unc.npz")

    # conditional model
    key, modul_cond_key = jrandom.split(key)

    model = GraphDiffusionAutoencoder(modul_cond_key, nlayer=nlayer, dim=dim)
    model = model.load_leaves("model10cond.npz")

    # model_fixed = functools.partial(model, adjacency=adjacency)

    # score function

    sample = jax.jit(sample, static_argnums=(1, 5, 6))

    min_sigma = sigma_lower_bound(n_atoms)
    max_sigma = sigma_upper_bound(n_atoms)

    min_sigma = jnp.log(min_sigma)
    max_sigma = jnp.log(max_sigma)

    step_size = 1e-3
    n_sigmas = 32

    sigmas = jnp.logspace(
        min_sigma,
        max_sigma,
        n_sigmas,
        base=jnp.e,
    )[::-1]

    num_iterations = 2048

    tempture = 0.8
    for weight in [0.7, 0.9, 1, 1.1, 1.3, 1.5]:
        conditional_probability = to_conditianal_probability(
            model_unc=model_unconditional,
            model_cond=model,
            encouding=adjacency,
            weight=1.2,
        )
        score = score_function(model_unconditional)
        score = jax.vmap(score)
        score = jax.jit(score)
        samples = sample(
            sigmas=sigmas,
            score=score,
            key=key,
            n_atoms=n_atoms,
            batch_size=batch_size,
            num_iterations=num_iterations,
            step_size=step_size,
            tempture=tempture,
        )

        plot = evaluate(
            samples=samples,
            max_degree=max_degree,
            file_name="test_symmetric" + str(tempture),
            use_wandb=True,
        )

        wandb.log(
            {
                "num_iterations": num_iterations,
                "step_size": step_size,
                "n_sigmas": n_sigmas,
                "tempture": tempture,
                "plot": plot,
            }
        )

    wandb.finish()
