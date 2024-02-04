import jax
import jax.random as jrandom
import numpy as np
import memmpy
import optax
import tqdm
import wandb
import datetime
import matplotlib.pyplot as plt

from data import gdb13_graph_memmap
from models import GraphDiffusionAutoencoder
from sklearn.decomposition import PCA


def main(
    model_path: str = "model.npz",
    natoms: int = 10,
    nlayer: int = 3,
    dim: int = 128,
    batch_size: int = 1024,
    seed: int = 0,
):
    data = gdb13_graph_memmap("data", natoms)

    data_valid = memmpy.split(data, "train", shuffle=True, seed=seed)  # type: ignore
    data_valid = memmpy.unwrap(data_valid)

    model = GraphDiffusionAutoencoder(
        jrandom.PRNGKey(0),
        nlayer=nlayer,
        dim=dim,
    )
    model = model.load_leaves(model_path)
    enoder = jax.jit(jax.vmap(model.encoder))

    encoded = []
    for batch in tqdm.tqdm(memmpy.Batched(data_valid, batch_size, False)):
        encoded.append(enoder(batch))

    encoded = np.concatenate(encoded, axis=0)

    pca = PCA(n_components=2)
    pca.fit(encoded)

    encoded = pca.transform(encoded)

    plt.scatter(encoded[:, 0], encoded[:, 1])
    plt.savefig("encoded.png", dpi=300)

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    # wandb.init(
    #     project="sampling",
    #     config={
    #         "natoms": natoms,
    #         "seed": seed,
    #         "start time": timestamp,
    #     },
    # )

    # wandb.log({"encoded": wandb.Image("encoded.png")})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.npz")
    parser.add_argument("--natoms", type=int, default=10)
    parser.add_argument("--nlayer", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        model_path=args.model,
        natoms=args.natoms,
        nlayer=args.nlayer,
        dim=args.dim,
        batch_size=args.batch_size,
        seed=args.seed,
    )
