import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import angular_loss
from quasi.quasi_model import eval_quasi
import quasi.eval_image as eval_image
import quasi.model as model

class AngularGANv3Model(BaseModel):
    def name(self):
        return 'AngularGAN3Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Angular', type=float, default=1.0, help='influence of angular loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN',  'G_L1', 'G_Ang', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.quasimodel = "model.pt"
        
        model_data = torch.load(self.quasimodel.format(fold=fold))
        input_code = model_data["args"].input
        output_code = model_data["args"].output
        self.quasi_net = model.CCNet(input_code=input_code, output_code=output_code, noise=0.0, mask_clipped=args.mask_clipped)
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
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.quasi_B = self.quasimodel.process_img(self.real_A)
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        self.eps = torch.tensor(1e-04).to(self.device)
        real_AB = torch.cat((self.real_A, torch.div(self.real_A, torch.max(self.real_B, self.eps))), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.eps = torch.tensor(1e-04).to(self.device)
        self.loss_G_L1 = self.criterionL1(self.fake_B,   torch.div(self.real_A,  torch.max(self.real_B, self.eps))) * self.opt.lambda_L1 * 100
        
        self.illum_gt = self.real_B
        self.illum_pred = torch.div(self.real_A,  torch.max(self.fake_B, self.eps))
        self.loss_G_Ang = self.criterionAngular(self.illum_gt, self.illum_pred) * self.opt.lambda_Angular
        self.loss_quasi_Ang = self.quasi.process_img(self.illum_gt)
        self.loss_G = self.loss_G_GAN + self.loss_quasi_Ang + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_Ang + self.loss_G_L1

        self.loss_G.backward()

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
