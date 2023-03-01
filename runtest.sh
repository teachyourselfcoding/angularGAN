a=$(ls datasets/facades/test | wc -l)
echo $a
python test.py --dataroot ./datasets/NUS-8/test/A_reduced --name angular_gan --model angular_gan --which_direction AtoB --display_id -1 --how_many 5 --which_model_netG unet_256


python test.py --dataroot ./datasets/NUS-8/test/ --name angular_gan --model angular_gan --which_direction AtoB --display_id -1 --how_many 5 --which_model_netG unet_256

