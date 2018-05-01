##### Convert dataset to tfrecord format. #####
python datasets/multispectrum.py \
  --dataset multispectrum-vis \
  --root data/multispectrum/images \
  --outdir /tmp/data/img2img \
  --mode train-day  \
  --num_samples 3000

python datasets/multispectrum.py \
  --dataset multispectrum-lwir \
  --root data/multispectrum/images \
  --outdir /tmp/data/img2img \
  --mode train-day  \
  --num_samples 3000


###### Train UNIT ######
python unit/train.py \
  --arch unit \
  --outdir output/test \
  --x_tfrecord /tmp/data/img2img/multispectrum-vis-train-day-3000.tfrecord \
  --y_tfrecord /tmp/data/img2img/multispectrum-lwir-train-day-3000.tfrecord \
  --batch_size 1 \
  --target_height 256 \
  --target_width 256 \
  --snapshot 20000 \
  --iteration 100000 \
  --num_res_block 4 \
  --generator_lr 1e-4 \
  --discriminator_lr 1e-1 \
  --beta1 0.5 \
  --lambda_gan 10. \
  --lambda_vae_kl 0.1 \
  --lambda_rec 100. \
  --lambda_cycle_kl 0.1 \
  --lambda_cycle 100. \
  --ls_loss True


###### Convert test dataset to tfrecord format. ######
python datasets/multispectrum.py \
  --dataset multispectrum-vis \
  --root data/multispectrum/images \
  --outdir /tmp/data/img2img \
  --mode test-day  \
  --num_samples 300

python datasets/multispectrum.py \
  --dataset multispectrum-lwir \
  --root data/multispectrum/images \
  --outdir /tmp/data/img2img \
  --mode test-day  \
  --num_samples 300


###### Evaluate UNIT ######
dirname=unit-20180424-0255-26940
python unit/eval.py \
  --use_gpu_server True \
  --arch unit \
  --outdir output/test/${dirname}/results \
  --checkpoint_dir output/test/${dirname}/trained_model/model-15 \
  --x_tfrecord /tmp/data/img2img/multispectrum-vis-test-day-300.tfrecord \
  --y_tfrecord /tmp/data/img2img/multispectrum-lwir-test-day-300.tfrecord \
  --batch_size 1 \
  --target_height 256 \
  --target_width 256 \
  --num_sample 100