import jax
import jax.random as jrandom
import memmpy
import optax
import tqdm
import wandb
import datetime

from data import gdb13_graph_memmap
from models import GraphDiffusionAutoencoder, score_interpolation_loss_ae


def main(
    model_path: str = "model.npz",
    natoms: int = 10,
    nlayer: int = 3,
    dim: int = 128,
):
    model = GraphDiffusionAutoencoder(
        jrandom.PRNGKey(0),
        nlayer=nlayer,
        dim=dim,
    )
    model = model.load_leaves(model_path)

    enoder = jax.jit(model.encoder)
