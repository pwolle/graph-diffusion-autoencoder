import os
import tarfile

import memmpy
import wget
import itertools
from rdkit import Chem

import tqdm
from typeguard import typechecked

import tempfile


@typechecked
def countlines(path: str) -> int:
    return sum(1 for _ in open(path))


@typechecked
def gdb13_graphs(path: str, natoms: int = 7):
    """
    Generate binary adjacency matricees from smiles files.

    Parameters
    ---
    path: str
        Path to smiles files (wildcards with * can be set)

    files:
        Files to be used (how many atoms per molecule)
    """
    # Check if file exist, if not try to download
    path1 = os.path.join(path, f"{natoms}.g.smi")
    path2 = os.path.join(path, f"{natoms}.sk.smi")

    if not os.path.exists(path1) or not os.path.exists(path2):
        print("GDB13 files not found, downloading ...")

        # Download data
        tempdir = tempfile.gettempdir()

        wget.download(
            "https://zenodo.org/record/5172018/files/gdb13.g.tgz",
            out=tempdir,
        )
        wget.download(
            "https://zenodo.org/record/5172018/files/gdb13.sk.tgz",
            out=tempdir,
        )

        # Extract files
        file = tarfile.open(os.path.join(tempdir, "gdb13.g.tgz"))
        file.extractall(path)
        file.close()

        file = tarfile.open(os.path.join(tempdir, "gdb13.sk.tgz"))
        file.extractall(path)
        file.close()

        print("Download completed.")
    else:
        print("GDB13 files found.")

    lines1, lines2 = countlines(path1), countlines(path2)
    print(f"Reading {lines1 + lines2} SMILES from {path1} and {path2} ...")

    bar = tqdm.tqdm(
        itertools.chain(open(path1), open(path2)),
        total=lines1 + lines2,
    )

    for line in bar:
        mol = Chem.MolFromSmiles(line)  # type: ignore
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True)  # type: ignore
        yield adj.astype(bool)


def gdb13_graph_memmap(path: str, natoms: int = 7):
    path_smiles = os.path.join(path, "gdb13_smiles")
    path_memmap = os.path.join(path, "gdb13_memmap")

    name = f"gd13_adjacencies_natoms={natoms}"
    if name in memmpy.safe_load(path_memmap)["arrays"]:
        print("Using existing memmap file.")
        return memmpy.read_vector(path_memmap, name)

    print(f"GDB13 {natoms}-atom memmap not found, creating ...")

    with memmpy.WriteVector(path_memmap, name) as memfile:
        for adj in gdb13_graphs(path_smiles, natoms):
            memfile.append(adj)

    return memmpy.read_vector(path_memmap, name)


def _test():
    data = gdb13_graph_memmap("data", 10)
    print(data.shape)
    # for adjacency in gdb13_graphs("data", 13):
    #     # print(repr(adjacency))
    #     # break
    #     pass

    # from tqdm import tqdm

    # import time

    # # Create a range of values for the loop
    # for i in tqdm(range(10), desc="Processing"):
    #     # Simulate some work
    #     time.sleep(0.5)

    #     # Update the description during the loop
    #     tqdm.set_description(f"Processing item {i}")

    # # The loop is complete
    # print("Loop finished!")


if __name__ == "__main__":
    _test()
