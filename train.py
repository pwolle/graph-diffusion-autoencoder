import jax
import jax.random as jrandom
import memmpy
import optax
import tqdm
import wandb
import datetime

from data import gdb13_graph_memmap
from models import BinaryEdgesModel, score_interpolation_loss


def main(
    natoms: int = 10,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    nlayer: int = 3,
    dim: int = 128,
    dim_at: int = 128,
    seed: int = 0,
):
    key = jrandom.PRNGKey(seed)
    key, model_key = jrandom.split(key)

    print("Initializing model ...")
    model = BinaryEdgesModel(
        model_key,
        nlayer=nlayer,
        dim=dim,
        dim_at=dim_at,
    )

    optimizer = optax.adam(lr)
    optimizer_state = optimizer.init(model)  # type: ignore
    print("Model initialized.")

    @jax.jit
    def loss_fn(key, adjacencies, model):
        return score_interpolation_loss(key, adjacencies, jax.vmap(model))

    @jax.jit
    def train_step(key, adjacencies, model, optimizer_state):
        grad_fn = jax.value_and_grad(loss_fn, argnums=2)
        loss, grad = grad_fn(key, adjacencies, model)

        updates, optimizer_state = optimizer.update(grad, optimizer_state)
        model = optax.apply_updates(model, updates)
        return loss, model, optimizer_state

    print("Loading data ...")
    data = gdb13_graph_memmap("data", natoms)

    data_train = memmpy.split(data, "train", shuffle=True, seed=seed)  # type: ignore
    data_train = memmpy.Batched(data_train, batch_size, True)

    data_valid = memmpy.split(data, "valid", shuffle=True, seed=seed)  # type: ignore
    data_valid = memmpy.unwrap(data_valid)[:1024]

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="graph-diffusion-autoencoder",
        config={
            "natoms": natoms,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "nlayer": nlayer,
            "dim": dim,
            "dim_at": dim_at,
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
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dim_at", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        natoms=args.natoms,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        nlayer=args.nlayer,
        dim=args.dim,
        dim_at=args.dim_at,
        seed=args.seed,
    )
