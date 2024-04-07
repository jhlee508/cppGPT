# C++GPT

GPT-2 small inference implementation from scratch using C++.

FYI, tokenizer is not implemented.


## Setup

```bash
mkdir assets

# Download OpenAI's GPT2-small model parameter (125M)
python tools/download_gpt2_model.py
python tools/dump_tf_model.py 
```

## Run

```bash
make
make run # this executes run.sh
``` 