#!/bin/bash

TEST_BATCH_SIZE=10
BASE_MODEL=$1
LORA_WEIGHTS=$2

CURRENT_FOLDER=$(dirname $(readlink -f "$0"))
cd $CURRENT_FOLDER
bash run_base.sh v7b $TEST_BATCH_SIZE $BASE_MODEL cuda:0 &
bash run_lora.sh law $TEST_BATCH_SIZE $BASE_MODEL $LORA_WEIGHTS cuda:1 &
wait
cd ..
ecs collect_ratings.ecs ./output.csv ./test_law/ ./test_v7b/
zip -r ./output.zip ./output.csv ./test_law/ ./test_v7b/
rm -rf ./output.csv ./test_law/ ./test_v7b/