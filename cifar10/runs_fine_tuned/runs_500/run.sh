#!/bin/bash
python logreg_train_fine_tune.py \
    --num-train-images=500 \
    --random-subset \
    --logdir=./runs_fine_tuned/runs_500
