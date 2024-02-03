from sample_symmetric import sample, score_function
from models import BinaryEdgesModel
from sigma_intevall import sigma_lower_bound, sigma_upper_bound
from evaluate import evaluate

import jax
import jax.random as jrandom
import jax.numpy as jnp
import wandb
import datetime

if __name__ == "__main__":
    n_atoms = 11
    seed = 0
    batch_size = 1
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

    key = jrandom.PRNGKey(seed)
    key, modul_key = jrandom.split(key)
    key, eval_key = jrandom.split(key)

    model = BinaryEdgesModel(modul_key, nlayer=2, dim=dim)
    model = model.load_leaves("model_2024-01-19 01:44:18.npz")

    score = score_function(model)
    score = jax.vmap(score)
    score = jax.jit(score)

    sample = jax.jit(sample, static_argnums=(1, 4, 5))

    min_sigma = sigma_lower_bound(n_atoms)
    max_sigma = sigma_upper_bound(n_atoms)

    min_sigma = jnp.log(min_sigma)
    max_sigma = jnp.log(max_sigma)

    step_size = 1e-4
    n_sigmas = 20

    sigmas = jnp.logspace(
        min_sigma,
        max_sigma,
        n_sigmas,
        base=jnp.e,
    )[::-1]

    for num_iterations in [100, 300, 500, 8000, 1000]:
        samples = sample(
            sigmas=sigmas,
            score=score,
            key=key,
            n_atoms=n_atoms,
            batch_size=batch_size,
            num_iterations=num_iterations,
            step_size=step_size,
        )

        plot = evaluate(
            samples=samples,
            max_degree=max_degree,
            file_name="test_symmetric" + num_iterations,
            use_wandb=True,
        )

        wandb.log(
            {
                "num_iterations": num_iterations,
                "step_size": step_size,
                "n_sigmas": n_sigmas,
                "plot": plot,
            }
        )

    wandb.finish()
