import itertools
import os
import tarfile
import tempfile

import memmpy
import numpy as np
import tqdm
import wget
from rdkit import Chem
from typeguard import typechecked
from typing import Generator


@typechecked
def countlines(path: str) -> int:
    return sum(1 for _ in open(path))


@typechecked
def gdb13_graphs(
    path: str,
    natoms: int = 7,
) -> Generator[np.ndarray, None, None]:
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

    nlines = countlines(path1) + countlines(path2)
    print(f"Reading {nlines} SMILES from {path1} and {path2} ...")

    bar = tqdm.tqdm(itertools.chain(open(path1), open(path2)), total=nlines)
    for line in bar:
        mol = Chem.MolFromSmiles(line)  # type: ignore
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True)  # type: ignore
        yield adj.astype(bool)


def gdb13_graph_memmap(path: str, natoms: int = 7) -> np.memmap:
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
