from read_smiles import save_with_memmpy, adj_gen

# main
if __name__ == "__main__":
    data_path = "data/gdb13.g.tgz"

    gdb_files = "https://zenodo.org/record/5172018/files/gdb13.g.tgz"
    # Save adj matrices to memmap files
    save_with_memmpy(data_path, gdb_files)

    # Generate adj matrices from smiles files
    # adj_matrices = adj_gen(data_path, gdb_files)

    # Print first adj matrix
    # adj = next(adj_matrices)
    # print(adj)
