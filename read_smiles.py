import glob
from rdkit import Chem
import wget
import tarfile
import os

def adj_gen(path,files):
    """Generate adj matricees from smiles files (Generator function) 
    
    :param path: Path to directory where the smiles files are stored (Wildcard can be used)
    :param files: Files to be used (Wildcards can be used)
    :return:
    """
    #Check if file exist, if not try to download
    if not glob.glob(path + "/" +files):
        print(f"FILES DOES NOT EXIST YET, DOWNLOADING GDB files!")

        #Download data
        wget.download("https://zenodo.org/record/5172018/files/gdb13.g.tgz")
        wget.download("https://zenodo.org/record/5172018/files/gdb13.sk.tgz")

        #Extract files
        file = tarfile.open("gdb13.g.tgz")
        file.extractall(path)
        file.close()

        file = tarfile.open("gdb13.sk.tgz")
        file.extractall(path)
        file.close()

        #Delete tar files
        os.remove("gdb13.g.tgz")
        os.remove("gdb13.sk.tgz")
    
    # Read SMILES from the file
    for file_path in glob.glob(path + "/" + files):
        with open(file_path, 'r') as file:
            for line in file:
                mol = Chem.MolFromSmiles(line)
                yield Chem.GetAdjacencyMatrix(mol)
