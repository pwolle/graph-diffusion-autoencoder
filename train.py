def main(
    natoms: int = 10,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    nlayer: int = 3,
    seed: int = 0,
    hidden_dim: int = 128,
    attention_dim: int = 128,
):
    from models import BinaryEdgesModel, score_interpolation_loss
    from data import gdb13_graph_memmap

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--natoms", type=int, default=10)
    args = parser.parse_args()

    main(
        natoms=args.natoms,
    )
