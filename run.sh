
python train.py --checkpoints_dir ./checkpoints/checkpoint_v2 --dataroot ./datasets/NUS-8/train/A --name angular_gan --model angular_gan_v3 --lambda_Angular 1  --lambda_L1 1 --which_model_netG unet_256  --dataset_mode single --no_lsgan --norm batch --pool_size 0 --save_epoch_freq 10  --niter 200 --continue_train --which_epoch latest  --which_model_netG unet_256

