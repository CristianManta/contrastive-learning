#!/bin/bash

# Shell script to launch PyTorch model testing experiments.
# This is a template to use as a starting point for creating
# your own script files.

# Instructions for use:
# Make sure the paths are correct and execute from the command
# line using:
# $ ./yourscript.sh
# You will have to change file permissions before you can
# execute it:
# $ chmod +x yourscript.sh
# To automate the execution of multiple scipts use the
# jobdispatcher.py tool.

MODEL='ResNet50'

# Setup
DATASET='cifar10'   # Use the dataset name in LOGDIR
DATADIR='/home/math/oberman-lab/data/cifar10'  # Shared data file store

# If you want to specify which GPU to run on,
# prepend the following with
#CUDA_VISIBLE_DEVICES=<id> \
# or alternately, from the command line issue
# $ export CUDA_VISIBLE_DEVICES=<id>
# to make only that GPU visible
python train_baseline_cfinlay.py \
    --bn \
    --lr 0.1 \
    --lr-schedule '[[0,1],[60,0.2],[120,0.04],[160,0.008]]'\
    --cutout 0 \
    --epochs 200 \
    --test-batch-size 100 \
    --model $MODEL \
    --dataset $DATASET \
    --datadir $DATADIR
