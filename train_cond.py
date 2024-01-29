import jax
import jax.random as jrandom
import memmpy
import optax
import tqdm
import wandb
import datetime

from data import gdb13_graph_memmap
from models import BinaryEdgesModel
from models_cond import BinaryEdgesModel_cond, score_interpolation_loss_cond, Encoder

def main(
    natoms: int = 10,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    nlayer: int = 3,
    dim: int = 128,
    seed: int = 0,
    dim_latent: int = 1024,
    model_path: str = "",
):

    print("Loading data ...")
    data = gdb13_graph_memmap("data", natoms)

    data_train = memmpy.split(data, "train", shuffle=True, seed=seed)  # type: ignore
    data_train = memmpy.Batched(data_train, batch_size, True)

    data_valid = memmpy.split(data, "valid", shuffle=True, seed=seed)  # type: ignore
    data_valid = memmpy.unwrap(data_valid)[:1024 * 4]

    key = jrandom.PRNGKey(seed)
    key, model_key = jrandom.split(key)

    print("Initializing model ...")
    model_cond = BinaryEdgesModel_cond(
        key=model_key,
        nlayer=nlayer,
        dim=dim,
        dim_latent=dim_latent,
    )
    
    model = BinaryEdgesModel(
        model_key,
        nlayer=nlayer,
        dim=dim,
    )

    if model_path != "":
        model = model.load_leaves(model_path)

    # optimizer = optax.adam(lr)
    # optimizer_state = optimizer.init(model)  # type: ignore
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=50,
        decay_steps=len(data_train) * epochs,
        end_value=1e-6,
    )

    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-5)
    optimizer_state = optimizer.init(model_cond)  # type: ignore
    print("Model initialized.")

    @jax.jit
    def loss_fn(key, adjacencies, model, model_cond):
        return score_interpolation_loss_cond(key,
                                             adjacencies,
                                             jax.vmap(model),
                                             jax.vmap(model_cond),
                                             )

    @jax.jit
    def train_step(key, adjacencies, model, model_cond, optimizer_state):
        grad_fn = jax.value_and_grad(loss_fn, argnums=3)
        loss, grad = grad_fn(key, adjacencies, model, model_cond)

        updates, optimizer_state = optimizer.update(
            grad,
            optimizer_state, 
            model_cond,
        )
        model_cond = optax.apply_updates(model_cond, updates)
        return loss, model_cond, optimizer_state

    print("Loading data ...")

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
            "seed": seed,
            "start time": timestamp,
        },
    )

    for epoch in range(epochs):
        print(f"Starting epoch {epoch} ...")
        for train_batch in tqdm.tqdm(data_train):
            train_batch = memmpy.unwrap(train_batch)

            key, train_key = jrandom.split(key)
            loss_train, model_cond, optimizer_state = train_step(
                train_key, train_batch, model, model_cond, optimizer_state
            )

            loss_valid = loss_fn(key, data_valid, model, model_cond)
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
    parser.add_argument("--natoms", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim_latent", type=int, default=1024)
    parser.add_argument("--model_path", type=str, default="")
    args = parser.parse_args()

    main(
        natoms=args.natoms,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        nlayer=args.nlayer,
        dim=args.dim,
        seed=args.seed,
        dim_latent=args.dim_latent,
        model_path=args.model_path
    )
