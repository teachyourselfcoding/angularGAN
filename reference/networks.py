import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from math import pi
from functools import partial
from packaging import version
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed, Block

from models.transformer import MaskedAutoencoderViT
from models.uformer import Uformer, Uformer8
from models.eval_network import ProposedNetwork
from models.diffusion import Diffusion
from models.denoising_diffusion_pytorch import Unet_attn
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],aux_num=0):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # Contrast
    elif which_model_netG == 'unet_256_contrast':
        netG = Unet_bottleneck(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # Unet MTL
    elif which_model_netG == 'unet':
        netG = Unet(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_mtl':
        netG = Unet_mtl(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # UViT
    elif which_model_netG == 'uvit':
        netG = MaskedAutoencoderViT(
            img_size=256, patch_size=16, embed_dim=768, depth=8, num_heads=16,
            decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif which_model_netG == 'uvit_small':
        netG = MaskedAutoencoderViT(
            img_size=256, patch_size=16, embed_dim=768, depth=4, num_heads=12,
            decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=12,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif which_model_netG == 'uformer':
        netG = Uformer()
    elif which_model_netG == 'unet_diff':
        netG = Unet_attn(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8)
                )
    elif which_model_netG == 'unet_attn':
        netG = Unet_attn(
                    dim = 64,
                    dim_mults = (1, 2, 4, 8),
                    with_time_emb = False,
                    aux_num = aux_num
                )
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic': 
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'meta_net':
        netD = MetaNet()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, python train.py --model angular_gan_v2 --which_model_netG custom --dataroot ../datasets/NUS-8/testing/ --name nus8test,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# U-Net for contrastive learning
class Unet_bottleneck(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet_bottleneck, self).__init__()

        # construct unet structure
        #self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 3,256,256 -> 64,128,128
        self.down_block_1 = nn.Sequential(*[
            nn.Conv2d(input_nc, ngf, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
        ])
        # 64,128,128 -> 128,64,64
        self.down_block_2 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2)
        ])
        # 128,64,64 -> 256,32,32
        self.down_block_3 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4)
        ])
        # 256,32,32 -> 512,16,16
        self.down_block_4 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,16,16 -> 512,8,8
        self.down_block_5 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,8,8 -> 512,4,4
        self.down_block_6 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,4,4 -> 512,2,2
        self.down_block_7 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,2,2 -> 512,1,1
        self.down_block_8 = nn.Sequential(*[
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        ])

        # bottleneck 512,1,1 -> 512
        self.mlp = nn.Sequential(*[
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True)
        ])

        # 512,1,1 -> 512,2,2
        self.up_block_8 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8 = nn.Sequential(*self.up_block_8)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7 += [nn.Dropout(0.5)]
        self.up_block_7 = nn.Sequential(*self.up_block_7)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6 += [nn.Dropout(0.5)]
        self.up_block_6 = nn.Sequential(*self.up_block_6)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5 += [nn.Dropout(0.5)]
        self.up_block_5 = nn.Sequential(*self.up_block_5)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4 = nn.Sequential(*self.up_block_4)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3 = nn.Sequential(*self.up_block_3)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2 = nn.Sequential(*self.up_block_2)
        # (64+64),128,128 -> 3,256,256
        self.up_block_1 = [
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 2, output_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1 = nn.Sequential(*self.up_block_1)


    def forward(self, x, only_encoder=False):
        x_1 = self.down_block_1(x)      #64,128,128
        x_2 = self.down_block_2(x_1)    #128,64,64
        x_3 = self.down_block_3(x_2)    #256,32,32
        x_4 = self.down_block_4(x_3)    #512,16,16
        x_5 = self.down_block_5(x_4)    #512,8,8
        x_6 = self.down_block_6(x_5)    #512,4,4
        x_7 = self.down_block_7(x_6)    #512,2,2
        x_8 = self.down_block_8(x_7)    #512,1,1

        #m_lyr = x_8.clone() #512,1,1

        if only_encoder:
          m_lyr = self.mlp(x_8)

          return m_lyr
        
        else:
          
          x_8 = self.up_block_8(x_8)                 #512,2,2
          x_7 = self.up_block_7(torch.cat([x_8, x_7], 1))     #512,4,4
          x_6 = self.up_block_6(torch.cat([x_7, x_6], 1))     #512,8,8
          x_5 = self.up_block_5(torch.cat([x_6, x_5], 1))     #512,16,16
          x_4 = self.up_block_4(torch.cat([x_5, x_4], 1))     #256,32,32
          x_3 = self.up_block_3(torch.cat([x_4, x_3], 1))     #128,64,64
          x_2 = self.up_block_2(torch.cat([x_3, x_2], 1))     #64,128,128
          output = self.up_block_1(torch.cat([x_2, x_1], 1))    #3,256,256

          return output

# General U-Net
class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet, self).__init__()

        # construct unet structure
        #self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        # --------------------Encoder--------------------
        # 3,256,256 -> 64,128,128
        self.down_block_1 = nn.Sequential(*[
            nn.Conv2d(input_nc, ngf, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
        ])
        # 64,128,128 -> 128,64,64
        self.down_block_2 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2)
        ])
        # 128,64,64 -> 256,32,32
        self.down_block_3 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4)
        ])
        # 256,32,32 -> 512,16,16
        self.down_block_4 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,16,16 -> 512,8,8
        self.down_block_5 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,8,8 -> 512,4,4
        self.down_block_6 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,4,4 -> 512,2,2
        self.down_block_7 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,2,2 -> 512,1,1
        self.down_block_8 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        ])

        self.encoder = nn.ModuleList([
            self.down_block_1,
            self.down_block_2,
            self.down_block_3,
            self.down_block_4,
            self.down_block_5,
            self.down_block_6,
            self.down_block_7,
            self.down_block_8
        ])

        # bottleneck 512,1,1 -> 512
        self.mlp = nn.Sequential(*[
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True)
        ])

        # --------------------Decoder--------------------
        # 512,1,1 -> 512,2,2
        self.up_block_8 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8 = nn.Sequential(*self.up_block_8)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7 += [nn.Dropout(0.5)]
        self.up_block_7 = nn.Sequential(*self.up_block_7)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6 += [nn.Dropout(0.5)]
        self.up_block_6 = nn.Sequential(*self.up_block_6)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5 += [nn.Dropout(0.5)]
        self.up_block_5 = nn.Sequential(*self.up_block_5)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4 = nn.Sequential(*self.up_block_4)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3 = nn.Sequential(*self.up_block_3)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2 = nn.Sequential(*self.up_block_2)
        # (64+64),128,128 -> 3,256,256
        self.up_block_1 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 2, output_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1 = nn.Sequential(*self.up_block_1)

        self.decoder = nn.ModuleList([
            self.up_block_8,
            self.up_block_7,
            self.up_block_6,
            self.up_block_5,
            self.up_block_4,
            self.up_block_3,
            self.up_block_2,
            self.up_block_1
        ])

        # --------------------Aux Decoder--------------------
        # 512,1,1 -> 512,2,2
        self.up_block_8_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8_aux = nn.Sequential(*self.up_block_8_aux)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7_aux += [nn.Dropout(0.5)]
        self.up_block_7_aux = nn.Sequential(*self.up_block_7_aux)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6_aux += [nn.Dropout(0.5)]
        self.up_block_6_aux = nn.Sequential(*self.up_block_6_aux)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5_aux += [nn.Dropout(0.5)]
        self.up_block_5_aux = nn.Sequential(*self.up_block_5_aux)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4_aux = nn.Sequential(*self.up_block_4_aux)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3_aux = nn.Sequential(*self.up_block_3_aux)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2_aux = nn.Sequential(*self.up_block_2_aux)
        #* (64+64),128,128 -> 1,256,256
        self.up_block_1_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 2, output_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1_aux = nn.Sequential(*self.up_block_1_aux)

        self.decoder_aux = nn.ModuleList([
            self.up_block_8_aux,
            self.up_block_7_aux,
            self.up_block_6_aux,
            self.up_block_5_aux,
            self.up_block_4_aux,
            self.up_block_3_aux,
            self.up_block_2_aux,
            self.up_block_1_aux
        ])


    def forward(self, x, only_encoder=False):
        
        x_enc = [x]

        for i,blk in enumerate(self.encoder):
            x_enc.append(blk(x_enc[i]))

        #x_1 = self.down_block_1(x)      #64,128,128
        #x_2 = self.down_block_2(x_1)    #128,64,64
        #x_3 = self.down_block_3(x_2)    #256,32,32
        #x_4 = self.down_block_4(x_3)    #512,16,16
        #x_5 = self.down_block_5(x_4)    #512,8,8
        #x_6 = self.down_block_6(x_5)    #512,4,4
        #x_7 = self.down_block_7(x_6)    #512,2,2
        #x_8 = self.down_block_8(x_7)    #512,1,1

        m_lyr = x_enc[-1]
        if only_encoder:
          m_lyr = self.mlp(m_lyr)
          return m_lyr
        
        else:
          # Primary Decoder
          x_dec = m_lyr
          for i,blk in enumerate(self.decoder):
            if i == 0: 
                x_dec = blk(x_dec)
            else:
                x_dec = blk(torch.cat([x_dec, x_enc[-i-1]], 1))

          #x_8 = self.up_block_8(x_8)                 #512,2,2
          #x_7 = self.up_block_7(torch.cat([x_8, x_7], 1))     #512,4,4
          #x_6 = self.up_block_6(torch.cat([x_7, x_6], 1))     #512,8,8
          #x_5 = self.up_block_5(torch.cat([x_6, x_5], 1))     #512,16,16
          #x_4 = self.up_block_4(torch.cat([x_5, x_4], 1))     #256,32,32
          #x_3 = self.up_block_3(torch.cat([x_4, x_3], 1))     #128,64,64
          #x_2 = self.up_block_2(torch.cat([x_3, x_2], 1))     #64,128,128
          #output = self.up_block_1(torch.cat([x_2, x_1], 1))    #3,256,256

          # Auxiliary Decoder
          x_dec_aux = m_lyr
          for i,blk in enumerate(self.decoder_aux):
            if i == 0: 
                x_dec_aux = blk(x_dec_aux)
            else:
                x_dec_aux = blk(torch.cat([x_dec_aux, x_enc[-i-1]], 1))

          return x_dec, x_dec_aux


class Unet_mtl(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Unet_mtl, self).__init__()

        # construct unet structure
        #self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        # --------------------Encoder--------------------
        # 3,256,256 -> 64,128,128
        self.down_block_1 = nn.Sequential(*[
            nn.Conv2d(input_nc, ngf, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
        ])
        # 64,128,128 -> 128,64,64
        self.down_block_2 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2)
        ])
        # 128,64,64 -> 256,32,32
        self.down_block_3 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4)
        ])
        # 256,32,32 -> 512,16,16
        self.down_block_4 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,16,16 -> 512,8,8
        self.down_block_5 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,8,8 -> 512,4,4
        self.down_block_6 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,4,4 -> 512,2,2
        self.down_block_7 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 8)
        ])
        # 512,2,2 -> 512,1,1
        self.down_block_8 = nn.Sequential(*[
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        ])

        self.encoder = nn.ModuleList([
            self.down_block_1,
            self.down_block_2,
            self.down_block_3,
            self.down_block_4,
            self.down_block_5,
            self.down_block_6,
            self.down_block_7,
            self.down_block_8
        ])

        # bottleneck 512,1,1 -> 512
        self.mlp = nn.Sequential(*[
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True),
            nn.Linear(ngf* 8, ngf* 8, bias=True),
            nn.ReLU(True)
        ])

        # --------------------Decoder--------------------
        # 512,1,1 -> 512,2,2
        self.up_block_8 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8 = nn.Sequential(*self.up_block_8)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7 += [nn.Dropout(0.5)]
        self.up_block_7 = nn.Sequential(*self.up_block_7)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6 += [nn.Dropout(0.5)]
        self.up_block_6 = nn.Sequential(*self.up_block_6)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5 += [nn.Dropout(0.5)]
        self.up_block_5 = nn.Sequential(*self.up_block_5)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4 = nn.Sequential(*self.up_block_4)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3 = nn.Sequential(*self.up_block_3)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2 = nn.Sequential(*self.up_block_2)
        # (64+64),128,128 -> 3,256,256
        self.up_block_1 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 2, output_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1 = nn.Sequential(*self.up_block_1)

        self.decoder = nn.ModuleList([
            self.up_block_8,
            self.up_block_7,
            self.up_block_6,
            self.up_block_5,
            self.up_block_4,
            self.up_block_3,
            self.up_block_2,
            self.up_block_1
        ])

        # --------------------Aux Decoder--------------------
        # 512,1,1 -> 512,2,2
        self.up_block_8_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8_aux = nn.Sequential(*self.up_block_8_aux)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7_aux += [nn.Dropout(0.5)]
        self.up_block_7_aux = nn.Sequential(*self.up_block_7_aux)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6_aux += [nn.Dropout(0.5)]
        self.up_block_6_aux = nn.Sequential(*self.up_block_6_aux)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5_aux += [nn.Dropout(0.5)]
        self.up_block_5_aux = nn.Sequential(*self.up_block_5_aux)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4_aux = nn.Sequential(*self.up_block_4_aux)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3_aux = nn.Sequential(*self.up_block_3_aux)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2_aux = nn.Sequential(*self.up_block_2_aux)
        #* (64+64),128,128 -> 1,256,256
        self.up_block_1_aux = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 2, output_nc-2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1_aux = nn.Sequential(*self.up_block_1_aux)

        self.decoder_aux = nn.ModuleList([
            self.up_block_8_aux,
            self.up_block_7_aux,
            self.up_block_6_aux,
            self.up_block_5_aux,
            self.up_block_4_aux,
            self.up_block_3_aux,
            self.up_block_2_aux,
            self.up_block_1_aux
        ])

        # --------------------Aux Decoder2--------------------
        # 512,1,1 -> 512,2,2
        self.up_block_8_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        self.up_block_8_aux2 = nn.Sequential(*self.up_block_8_aux2)
        # (512+512),2,2 -> 512,4,4
        self.up_block_7_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_7_aux2 += [nn.Dropout(0.5)]
        self.up_block_7_aux2 = nn.Sequential(*self.up_block_7_aux2)
        # (512+512),4,4 -> 512,8,8
        self.up_block_6_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_6_aux2 += [nn.Dropout(0.5)]
        self.up_block_6_aux2 = nn.Sequential(*self.up_block_6_aux2)
        # (512+512),8,8 -> 512,16,16
        self.up_block_5_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 8,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 8)
                ]
        if use_dropout:
          self.up_block_5_aux2 += [nn.Dropout(0.5)]
        self.up_block_5_aux2 = nn.Sequential(*self.up_block_5_aux2)
        # (512+512),16,16 -> 256,32,32
        self.up_block_4_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 16, ngf * 4,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 4)
                ]
        self.up_block_4_aux2 = nn.Sequential(*self.up_block_4_aux2)
        # (256+256),32,32 -> 128,64,64
        self.up_block_3_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 8, ngf * 2,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf * 2)
                ]
        self.up_block_3_aux2 = nn.Sequential(*self.up_block_3_aux2)
        # (128+128),64,64 -> 64,128,128
        self.up_block_2_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 4, ngf,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    norm_layer(ngf)
                ]
        self.up_block_2_aux2 = nn.Sequential(*self.up_block_2_aux2)
        #* (64+64),128,128 -> 1,256,256
        self.up_block_1_aux2 = [
                    nn.ReLU(),
                    nn.ConvTranspose2d(ngf * 2, output_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias),
                    nn.Tanh()
                ]
        self.up_block_1_aux2 = nn.Sequential(*self.up_block_1_aux2)

        self.decoder_aux2 = nn.ModuleList([
            self.up_block_8_aux2,
            self.up_block_7_aux2,
            self.up_block_6_aux2,
            self.up_block_5_aux2,
            self.up_block_4_aux2,
            self.up_block_3_aux2,
            self.up_block_2_aux2,
            self.up_block_1_aux2
        ])



    def forward(self, x, only_encoder=False):
        
        x_enc = [x]

        for i,blk in enumerate(self.encoder):
            x_enc.append(blk(x_enc[i]))

        #x_1 = self.down_block_1(x)      #64,128,128
        #x_2 = self.down_block_2(x_1)    #128,64,64
        #x_3 = self.down_block_3(x_2)    #256,32,32
        #x_4 = self.down_block_4(x_3)    #512,16,16
        #x_5 = self.down_block_5(x_4)    #512,8,8
        #x_6 = self.down_block_6(x_5)    #512,4,4
        #x_7 = self.down_block_7(x_6)    #512,2,2
        #x_8 = self.down_block_8(x_7)    #512,1,1

        m_lyr = x_enc[-1]
        if only_encoder:
          m_lyr = self.mlp(m_lyr)
          return m_lyr
        
        else:
          # Primary Decoder
          x_dec = m_lyr
          for i,blk in enumerate(self.decoder):
            if i == 0: 
                x_dec = blk(x_dec)
            else:
                x_dec = blk(torch.cat([x_dec, x_enc[-i-1]], 1))

          # Auxiliary Decoder
          x_dec_aux = m_lyr
          for i,blk in enumerate(self.decoder_aux):
            if i == 0: 
                x_dec_aux = blk(x_dec_aux)
            else:
                x_dec_aux = blk(torch.cat([x_dec_aux, x_enc[-i-1]], 1))
          
          # Auxiliary Decoder 2
          x_dec_aux2 = m_lyr
          for i,blk in enumerate(self.decoder_aux2):
            if i == 0: 
                x_dec_aux2 = blk(x_dec_aux2)
            else:
                x_dec_aux2 = blk(torch.cat([x_dec_aux2, x_enc[-i-1]], 1))

          return x_dec, x_dec_aux, x_dec_aux2

# Defines the Uncertainty loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
def EU_L1_loss(input, target, var=1.0):
    var = torch.clamp(var, min=0.001)
    s = torch.log(var)
    abs_diff = torch.abs(input-target)
    L_EU = torch.mean(torch.norm(abs_diff,dim=(1),p=1)*torch.exp(-s)+var)
    return L_EU


def EU_angular_loss(input, target, var):
    var = torch.clamp(var, min=0.001)
    s = torch.log(var)
    
    #calculate angular loss
    cos_between = torch.nn.CosineSimilarity(dim=1)
    cos = cos_between(target, input)
    cos = torch.clamp(cos,-0.99999, 0.99999)
    loss_map = torch.acos(cos) * 180 / pi

    abs_diff = loss_map
    L_EU = torch.mean(torch.norm(abs_diff,dim=(1),p=1))
    return L_EU


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        #if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
        #    batch_dim_for_bmm = 1
        #else:
        #    batch_dim_for_bmm = self.opt.batch_size
        batch_dim_for_bmm = self.opt.batchSize

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    

class SRCLoss(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.ConsineSimilarity = torch.nn.CosineSimilarity(dim=dim)
        self.Criterion = torch.nn.L1Loss()
    
    def forward(self, feat_t_, feat_s_):
        anchor_s = feat_s_[0].view(1,-1).repeat(64,1) # 64,512
        anchor_t = feat_t_[0].view(1,-1).repeat(64,1) # 64,512
        src_s = self.ConsineSimilarity(anchor_s, feat_s_)
        src_t = self.ConsineSimilarity(anchor_t, feat_t_)

        return self.Criterion(src_s, src_t)