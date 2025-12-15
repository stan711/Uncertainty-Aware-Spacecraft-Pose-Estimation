#! /usr/bin/env bash

################################################################################
# ODR (Single)
################################################################################

PRETRAIN=outputs/efficientdet_d3/full_config_selftrain/model_best.pth.tar
# PRETRAIN=outputs/efficientdet_d6/full_config_gn8/model_best.pth.tar

EXP='experiments/odr_phi3_B4_N1024.yaml'
EXP_NAME='odr/selftrain_phi3_B4_N1024'

for DOMAIN in lightbox sunlamp; do

    CSV=$DOMAIN/labels/test.csv

    NUM_SAMPLES=1024
    BATCH_SIZE=4

    python3 tools/odr.py --cfg $EXP VERBOSE True \
        MODEL.PRETRAIN_FILE $PRETRAIN \
        EXP_NAME $EXP_NAME \
        TRAIN.TRAIN_CSV $CSV TRAIN.VAL_CSV $CSV \
        ODR.NUM_TRAIN_SAMPLES $NUM_SAMPLES \
        ODR.IMAGES_PER_BATCH $BATCH_SIZE \
        SEED 2021

done