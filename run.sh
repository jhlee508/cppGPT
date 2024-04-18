#!bin/sh

mpirun --bind-to none -mca btl ^openib -npernode 4 \
    --oversubscribe -quiet \
    ./main \
        -i ./data/input.bin \
        -a ./data/answer.bin \
        -o ./data/output.bin \
        -v -s \
        -t 5 \
        -n 1 \
        # -w \