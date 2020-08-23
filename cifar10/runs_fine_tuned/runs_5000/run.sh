#!/bin/bash
python logreg_train_fine_tune.py \
    --num-train-images=5000 \
    --random-subset \
    --logdir=./runs_fine_tuned/runs_5000
