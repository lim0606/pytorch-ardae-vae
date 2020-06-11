# AR-DAE: Towards Unbiased Neural Entropy Gradient Estimation 
Pytorch implementation of AR-DAE on our paper: 
> Jae Hyun Lim, Aaron Courville, Christopher Pal, Chin-Wei Huang, *AR-DAE: Towards Unbiased Neural Entropy Gradient Estimation* (2020)

## Toy example of AR-DAE
Example code to train AR-DAE on swiss roll dataset:  
[ipython-notebook](https://github.com/lim0606/pytorch-ardae-vae/tree/master/notebooks/ardae_toy.ipynb)

## Energy function fitting with AR-DAE
Example code to train an implicit sampler using AR-DAE-based entropy gradient estimator:  
[ipython-notebook](https://github.com/lim0606/pytorch-ardae-vae/tree/master/notebooks/ardae_fit.ipynb)

## AR-DAE VAE
### Getting Started

#### Requirements
`python>=3.6`  
`pytorch==1.2.0`  
`tensorflow` (for tensorboardX)  
`tensorboardX`  

#### Dataset
```sh
# http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist
unzip data.zip -d .
```
#### Structure
- `data`: data folder
- `datasets`: dataloader definitions
- `models`: model definitions
- `utils`: miscelleneous functions
- `ivae_ardae.py`: main function to train model (ardae vae)
- `vae.py`: main function to train baselines (vae) 

### Experiments
#### Train
- For example, you can train an APoE model for vision and haptic data (# of modalities = 2) as follows,  
  ```sh
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
  ```  
  For more information, please find example scripts, `run_vae_25gaussians.sh`, `run_vae_dbmnist.sh`, and `run_vae_sbmnist.sh`.
  
## SAC-AR-DAE
please find the code at https://github.com/lim0606/pytorch-ardae-rl

## Contact
For questions and comments, feel free to contact [Jae Hyun Lim](mailto:jae.hyun.lim@umontreal.ca) and [Chin-Wei Huang](mailto:chin-wei.huang@umontreal.ca).

## License
MIT License

## Reference
```
@article{jaehyun2020ardae,
  title={{AR-DAE}: Towards Unbiased Neural Entropy Gradient Estimation},
  author={Jae Hyun Lim and
          Aaron Courville and
          Christopher J. Pal and
          Chin-Wei Huang},
  journal={arXiv preprint arXiv:2006.05164},
  year={2020}
}
```
