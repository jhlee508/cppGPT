# Usage: python print_output.py <input.bin> <output.bin>
# Executable generation: pyinstaller --onefile print_output.py

import sys
import numpy as np
from transformers import AutoTokenizer


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_file = sys.argv[1]  # get input file name from command 
    output_file = sys.argv[2]  # get output file name from command 

    # load input prompt and output from a binary file
    input = np.fromfile(input_file, dtype=np.int32)
    output = np.fromfile(output_file, dtype=np.int32)

    # load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Print input prompt and generated output
    # output shape is (Number of Prompts, Number of generated tokens)
    # input shape is (Number of Prompts, Length of input prompt (=10))
    N_SEQ = 10
    N_TOKENS = 5
    N_PROMPTS = len(output) // N_TOKENS
    for i in range(N_PROMPTS):
        print("Prompt #", i+1)

        # decode input prompt and generated output
        input_text = tokenizer.decode(input[i*N_SEQ:(i+1)*N_SEQ])
        output_text = tokenizer.decode(output[i*N_TOKENS:(i+1)*N_TOKENS])
        
        # print input prompt and generated output
        print(" Input Prompt:", input_text)
        print(" Generated Output:", output_text)
        print()  

    