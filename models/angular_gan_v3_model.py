import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import angular_loss
from models.quasi.eval_image import eval_image
import models.quasi.quasi_model as model
import models.quasi.ptcolormap as ptcolormap

from datetime import datetime
import util.util as util

class AngularGANv3Model(BaseModel):
    def name(self):
        return 'AngularGANv3Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='single')
        parser.set_defaults(which_model_netG='unet_256')
       
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Angular', type=float, default=1.0, help='influence of angular loss')
            

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'quasi_Ang']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'quasi_B', 'first_quasi_weight_map']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.quasimodel = "/home/yanfeng/angulargan/models/quasi/weights/ilsvrc12-eg.pt"
        fold=None
        loss_G_L1_multiplyer = torch.tensor(50.0, requires_grad=True)
        loss_quasi_Ang_multiplyer  = torch.tensor(50.0, requires_grad=True)
        # self.loss_G_L1_multiplyer = loss_G_L1_multiplyer.clamp(1, 100)
        # self.loss_quasi_Ang_multiplyer = loss_quasi_Ang_multiplyer.clamp(1, 100)

        self.loss_G_L1_multiplyer = 100
        self.loss_quasi_Ang_multiplyer = 100

        model_data = torch.load(self.quasimodel.format(fold=fold))
        input_code = model_data["args"].input
        output_code = model_data["args"].output
        self.quasi_net = model.CCNet(input_code=input_code, output_code=output_code, noise=0.0)
        self.quasi_net.load_state_dict(model_data["model"])
        self.quasimodel = eval_image(self.quasi_net)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionAngular = angular_loss.angular_loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def save_image(self, im):
        im = util.tensor2im(im)
        time_now = datetime.now()
        current_time = time_now.strftime("%H:%M:%S")
        util.save_image(im, "/home/yanfeng/angulargan/checkpoints/angular_gan/00-34-24_07/12/2022/"  + current_time +'.png')

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        self.quasi_B, self.quasi_weights= self.quasimodel.process_img(self.real_A)
        self.first_quasi_weight_map = ptcolormap.apply_map(self.quasi_weights, 0, 1).squeeze(2)
        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # pred_real = self.netD(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real = 0
        
        # Combined loss
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * 0
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) 
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.quasi_weights * self.fake_B, self.quasi_weights * self.quasi_B)* self.opt.lambda_L1
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 * 100 * 0
        
        self.eps = torch.tensor(1e-04).to(self.device)
        self.illum_gt = 0
        # self.illum_gt = torch.div(self.real_A, torch.max(self.real_B, self.eps))
        self.illum_pred = torch.div(self.real_A, torch.max(self.fake_B, self.eps))
        self.loss_G_Ang = 0
        # self.loss_G_Ang = self.criterionAngular(self.illum_gt, self.illum_pred) * self.opt.lambda_Angular
        self.loss_quasi_Ang = self.quasimodel.get_illum(self.fake_B, self.quasi_weights)
        self.loss_G_L1= self.loss_G_L1 * self.loss_G_L1_multiplyer
        self.loss_quasi_Ang = self.loss_quasi_Ang * self.loss_quasi_Ang_multiplyer
        self.loss_G =  self.loss_G_L1   + self.loss_quasi_Ang 
        
        # self.save_image(self.second_quasi_weight_map)
        # self.loss_G = self.loss_G_GAN + self.loss_quasi_Ang + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
