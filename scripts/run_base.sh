#!/bin/bash

TEST_NAME=$1
TEST_FOLDER=test_$TEST_NAME
TEST_BATCH_SIZE=$2
BASE_MODEL=$3
CUDA_DEVICE=$4

CURRENT_FOLDER=$(dirname $(readlink -f "$0"))
cd $CURRENT_FOLDER/..
mkdir $TEST_FOLDER
ecs extract_test_data.ecs $TEST_BATCH_SIZE $TEST_FOLDER/test_data.json
python eval.py --no_lora True --base_model $BASE_MODEL --device $CUDA_DEVICE \
    --input $TEST_FOLDER/test_data.json --output $TEST_FOLDER/eval_output.json
python rating.py --input $TEST_FOLDER/eval_output.json --output $TEST_FOLDER/rating_output.json
