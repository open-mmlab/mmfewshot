#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
TIMES=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/multi_test.py $CONFIG $TIMES --launcher pytorch ${@:4}
