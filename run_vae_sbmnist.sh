#!/bin/bash

# baseline models
# resconv
python vae.py --cache experiments/sbmnist --dataset sbmnist --nheight 28 --nchannels 1 --train-batch-size 128 --eval-batch-size 1 --optimizer adam --momentum 0.9 --beta1 0.9 --model resconv --model-z-dim 32 --model-h-dim 0 --model-n-layers 0 --model-nonlin elu --model-n-dim 0 --model-clip-logvar none --exp-num 1 --lr 0.001 --beta-init 0.0001 --beta-fin 1.0 --beta-annealing 50000 --epochs 6400 --eval-iws-interval 5000 --iws-samples 256 --weight-avg none --weight-avg-start -1 --weight-avg-decay 0.998 --log-interval 100 --vis-interval 10000 --ckpt-interval 5000 --train-mode train

python vae.py --cache experiments/sbmnist --dataset sbmnist --nheight 28 --nchannels 1 --train-batch-size 128 --eval-batch-size 1 --optimizer adam --momentum 0.9 --beta1 0.9 --model resconv --model-z-dim 32 --model-h-dim 0 --model-n-layers 0 --model-nonlin elu --model-n-dim 0 --model-clip-logvar none --exp-num 1 --lr 0.001 --beta-init 0.0001 --beta-fin 1.0 --beta-annealing 50000 --epochs 6400 --eval-iws-interval 5000 --iws-samples 256 --weight-avg none --weight-avg-start -1 --weight-avg-decay 0.998 --log-interval 100 --vis-interval 10000 --ckpt-interval 5000 --train-mode final


# proposed method
# implicit resconv
python ivae_ardae.py --cache experiments/sbmnist --dataset sbmnist --nheight 28 --nchannels 1 --train-batch-size 128 --eval-batch-size 1 --m-optimizer adam --m-momentum 0.9 --m-beta1 0.9 --d-optimizer rmsprop --d-momentum 0.9 --d-beta1 0.9 --train-nstd-cdae 1 --train-nz-cdae 625 --train-nz-model 1 --model resconvct-res --model-z-dim 32 --model-h-dim 512 --model-n-layers 1 --model-nonlin elu --model-n-dim 100 --model-clip-z0-logvar none --model-clip-z-logvar none --cdae mlp-res --cdae-h-dim 512 --cdae-n-layers 5 --cdae-nonlin softplus --cdae-ctx-type lt0 --exp-num 1 --m-lr 0.001 --d-lr 0.0001 --beta-init 0.0001 --beta-fin 1.0 --beta-annealing 50000 --delta 0.1 --std-scale 100 --num-cdae-updates 2 --epochs 6400 --eval-iws-interval 5000 --iws-samples 256 --m-weight-avg none --m-weight-avg-start -1 --m-weight-avg-decay 0.998 --log-interval 100 --vis-interval 10000 --ckpt-interval 800 --train-mode train

python ivae_ardae.py --cache experiments/sbmnist --dataset sbmnist --nheight 28 --nchannels 1 --train-batch-size 128 --eval-batch-size 1 --m-optimizer adam --m-momentum 0.9 --m-beta1 0.9 --d-optimizer rmsprop --d-momentum 0.9 --d-beta1 0.9 --train-nstd-cdae 1 --train-nz-cdae 625 --train-nz-model 1 --model resconvct-res --model-z-dim 32 --model-h-dim 512 --model-n-layers 1 --model-nonlin elu --model-n-dim 100 --model-clip-z0-logvar none --model-clip-z-logvar none --cdae mlp-res --cdae-h-dim 512 --cdae-n-layers 5 --cdae-nonlin softplus --cdae-ctx-type lt0 --exp-num 1 --m-lr 0.001 --d-lr 0.0001 --beta-init 0.0001 --beta-fin 1.0 --beta-annealing 50000 --delta 0.1 --std-scale 100 --num-cdae-updates 2 --epochs 6400 --eval-iws-interval 5000 --iws-samples 256 --m-weight-avg none --m-weight-avg-start -1 --m-weight-avg-decay 0.998 --log-interval 100 --vis-interval 10000 --ckpt-interval 800 --train-mode final