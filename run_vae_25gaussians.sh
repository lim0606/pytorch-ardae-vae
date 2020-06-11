#!/bin/bash
python ivae_ardae.py \
    --cache experiments/25gaussians \
    --dataset 25gaussians --nheight 1 --nchannels 2 \
    --model mlp-concat --model-z-dim 2 --model-h-dim 256 --model-n-layers 2 --model-nonlin relu --model-n-dim 10 --model-clip-z0-logvar none --model-clip-z-logvar none \
    --cdae mlp-grad --cdae-h-dim 256 --cdae-n-layers 3 --cdae-nonlin softplus --cdae-ctx-type lt0 \
    --train-batch-size 512 --eval-batch-size 1 --train-nz-cdae 256 --train-nz-model 1 \
    --delta 0.1 --std-scale 10000 --num-cdae-updates 1 \
    --m-lr 0.0001 --m-optimizer adam --m-momentum 0.5 --m-beta1 0.5 \
    --d-lr 0.0001 --d-optimizer rmsprop --d-momentum 0.5 --d-beta1 0.5 \
    --epochs 16 \
    --eval-iws-interval 0 --iws-samples 64 --log-interval 100 --vis-interval 100 --ckpt-interval 1000 --exp-num 1
