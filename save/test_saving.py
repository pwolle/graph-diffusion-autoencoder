from read_smiles import save_with_memmpy, adj_gen
import memmpy

# main
if __name__ == "__main__":
    data_path = "data"

    # Save adj matrices to memmap filess
    save_with_memmpy(data_path, "9.*")

    memmap_data = memmpy.read_vector(path="data/mmpy", name="adj")
    print(memmap_data[0])
