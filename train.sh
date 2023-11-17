#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

NUM_GPUS=$(($(echo $GPUS | grep -o ',' | wc -l) + 1)) # count number of ',' and plus one

echo "Config: $CONFIG"
echo "GPUS: $GPUS"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "PORT: $PORT"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NUM_GPUS: $NUM_GPUS"

MKL_NUM_THREADS=4 \
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=$GPUS \
PYTHONPATH="$(dirname $0)":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    tools/train.py $CONFIG > logs/monodetr.log
