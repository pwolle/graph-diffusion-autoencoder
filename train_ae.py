import jax
import jax.random as jrandom
import memmpy
import optax
import tqdm
import wandb
import datetime

from data import gdb13_graph_memmap
from models import BinaryEdgesModel, score_interpolation_loss_ae


def main(
    natoms: int = 10,
    batch_size: int = 64,
    epochs: int = 100,
    nlayer: int = 3,
    dim: int = 128,
    seed: int = 0,
):
    print("Loading data ...")
    data = gdb13_graph_memmap("data", natoms)

    data_train = memmpy.split(data, "train", shuffle=True, seed=seed)  # type: ignore
    data_train = memmpy.Batched(data_train, batch_size, True)

    data_valid = memmpy.split(data, "valid", shuffle=True, seed=seed)  # type: ignore
    data_valid = memmpy.unwrap(data_valid)[: 1024 * 4]

    key = jrandom.PRNGKey(seed)
    key, model_key = jrandom.split(key)

    print("Initializing model ...")
    model = BinaryEdgesModel(
        model_key,
        nlayer=nlayer,
        dim=dim,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=10,
        decay_steps=len(data_train) * epochs,
        end_value=1e-4,
    )

    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
    optimizer_state = optimizer.init(model)  # type: ignore
    print("Model initialized.")

    @jax.jit
    def loss_fn(key, adjacencies, model):
        return score_interpolation_loss_ae(key, adjacencies, jax.vmap(model))

    @jax.jit
    def train_step(key, adjacencies, model, optimizer_state):
        grad_fn = jax.value_and_grad(loss_fn, argnums=2)
        loss, grad = grad_fn(key, adjacencies, model)

        updates, optimizer_state = optimizer.update(
            grad,
            optimizer_state,
            model,
        )
        model = optax.apply_updates(model, updates)
        return loss, model, optimizer_state

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="graph-diffusion-autoencoder",
        config={
            "natoms": natoms,
            "batch_size": batch_size,
            "epochs": epochs,
            "nlayer": nlayer,
            "dim": dim,
            "seed": seed,
            "start time": timestamp,
        },
    )

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} ...")
        for train_batch in tqdm.tqdm(data_train):
            train_batch = memmpy.unwrap(train_batch)

            key, train_key = jrandom.split(key)
            loss_train, model, optimizer_state = train_step(
                train_key, train_batch, model, optimizer_state
            )

            loss_valid = loss_fn(key, data_valid, model)

            wandb.log(
                {
                    "loss_train": loss_train,
                    "loss_valid": loss_valid,
                }
            )

    model.save_leaves(f"model_{timestamp}.npz")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--natoms", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--nlayer", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        natoms=args.natoms,
        batch_size=args.batch_size,
        epochs=args.epochs,
        nlayer=args.nlayer,
        dim=args.dim,
        seed=args.seed,
    )
