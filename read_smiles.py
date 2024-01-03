import glob
from rdkit import Chem

def adj_gen(path, gdb_files):
    """Generate adj matricees from smiles files 
    
    :param path: Path to smiles files (wildcards with * can be set)
    :param gdb_files: Used if the smiles files dont exist yet --> Automatically downloads it 
    """
    
    #Check if file exist, if not try to download
    if not glob.glob(path):
        print(f"FILES DOES NOT EXIST YET, DOWNLOADING {gdb_files}!")
        #TO BE IMPLEMENTED: DOWNLOADING FILES AUTOMATICALLY
    
    #Check again if file exist now after downloading
    if not glob.glob(path):
        raise FileNotFoundError("FILES COULD NOT BE FOUND OR DOWNLOADED!!")
    
    # Read SMILES from the file
    for file_path in glob.glob(path):
        with open(file_path, 'r') as file:
            for line in file:
                mol = Chem.MolFromSmiles(line)
                yield Chem.GetAdjacencyMatrix(mol)