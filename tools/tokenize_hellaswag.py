# Usage: python script.py data/input.bin

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# main function
if __name__ == "__main__":
    print("Loading Huggingface Hellaswag dataset...")
    dataset = load_dataset("Rowan/hellaswag")

    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    all_tokenized_sequences = [] # store all tokenized sequences

    # number of sequences to tokenize
    num_sequences = pow(2, 14)
    short_sequences = 0
    long_sequences = 0

    for i in range(len(dataset["train"])):  
        sequence = dataset["train"][i]["ctx"]
        # print("Text:", sequence)

        # tokenize sequence
        tokenized_sequence = tokenizer(sequence)["input_ids"]
        if len(tokenized_sequence) < 10:
            short_sequences += 1
            # print("#{} tokenized sequence length is less than 10.".format(i)) 
        else:
            long_sequences += 1
            all_tokenized_sequences.append(tokenized_sequence[:10]) # store first 10 tokens
            # print("Huggingface tokenizer output:", tokenized_sequence[:10])
            # print()
    print("Number of sequences with less than 10 tokens:", short_sequences)
    print("Number of sequences with 10 or more tokens:", long_sequences)

    # Select a subset of sequences
    tokenized_sequences = all_tokenized_sequences[:num_sequences]

    # Save tokenized sequences to a binary file (.bin)
    tokenized_sequences_array = np.array(tokenized_sequences, dtype=np.int32)
    tokenized_sequences_array.tofile('./data/input16K.bin')
    
    print("{} tokenized sequences saved to '.bin' file.".format(len(tokenized_sequences)))