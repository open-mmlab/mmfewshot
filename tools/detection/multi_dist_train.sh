#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
TIMES=$3
SEED=${SEED:-2021}
START=${START:-0}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/multi_train.py $CONFIG $TIMES --seed $SEED --start $START \
    --launcher pytorch ${@:4}
