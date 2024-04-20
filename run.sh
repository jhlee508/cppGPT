#!bin/sh

mpirun --bind-to none -mca btl ^openib -npernode 1 \
    --oversubscribe -quiet \
    ./main \
    -v -s \
    -t 5 \
    -n 1 \
    # -w \