# Usage: python read_tokenized_binary.py <filename>

import sys
import numpy as np


# main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    file_name = sys.argv[1]  # get file name from command 

    # load data from a binary file
    data = np.fromfile('./' + file_name, dtype=np.int32)

    # print size of the binary file
    file_size = 0
    for i in range(len(data)):
        file_size += 1
    print("Size of the binary file:", file_size)

    # print first 10 elements of the binary file
    print("First 10 elements of the binary file:")
    for i in range(10):
        print(data[i], end=' ')
    print()
