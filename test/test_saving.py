from data import save_with_memmpy, gdb13_graphs
import memmpy

# main
if __name__ == "__main__":
    data_path = "data"

    # Save adj matrices to memmap filess
    save_with_memmpy(data_path, "9.*")

    memmap_data = memmpy.read_vector(path="data/mmpy", name="adj")
    print(memmap_data[0])
