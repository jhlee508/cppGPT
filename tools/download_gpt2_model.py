# Acknowledgement: This code is originated from https://github.com/keith2018/TinyGPT
# Description: This script downloads the GPT-2 small model from OpenAI
# Usage: python download_gpt2_model.py

import requests
from tqdm import tqdm


def download(url: str, fname: str, chunk_size=4096):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=4096,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = f"https://openaipublic.blob.core.windows.net/gpt-2/models/124M/{filename}"
        fname = f"./assets/{filename}"
        download(url, fname)