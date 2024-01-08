import glob
from rdkit import Chem
import memmpy


def adj_gen(path, gdb_files):
    """
    Generate adj matricees from smiles files

    Parameters
    ---
    path:
        Path to smiles files (wildcards with * can be set)

    gdb_files:
        Used if the smiles files dont exist yet --> Automatically downloads it
    """

    # Check if file exist, if not try to download
    if not glob.glob(path):
        print(f"FILES DOES NOT EXIST YET, DOWNLOADING {gdb_files}!")
        # TO BE IMPLEMENTED: DOWNLOADING FILES AUTOMATICALLY

    # Check again if file exist now after downloading
    if not glob.glob(path):
        raise FileNotFoundError("FILES COULD NOT BE FOUND OR DOWNLOADED!!")

    # Read SMILES from the file
    for file_path in glob.glob(path):
        with open(file_path, "r") as file:
            for line in file:
                mol = Chem.MolFromSmiles(line)
                yield Chem.GetAdjacencyMatrix(mol)


def save_with_memmpy(
    data_path: str,
    gdb_files,
    memmpy_path: str = "data.mmpy",
    key: str = "adj",
):
    """
    Save adj matrices to memmap files

    Parameters
    ---
    path: str
        Path to smiles files (wildcards with * can be set)

    gdb_files:
        Used if the smiles files dont exist yet --> Automatically downloads it

    memmpy_path: str
        Path to memmap file

    key: str
        Key to access the memmap file
    """

    dataset = adj_gen(data_path, gdb_files)
    with memmpy.WriteVector(path=memmpy_path, name=key) as memfile:
        for adj in dataset:
            memfile.append(adj)
