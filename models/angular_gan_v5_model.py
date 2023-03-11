import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import angular_loss
from models import create_model

class AngularGANModel(BaseModel):
    def name(self):
        return 'AngularGANModel'

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
        self.loss_names = ['G_GAN', 'G_L1', 'G_Ang', 'D_real', 'D_fake']
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
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.display_id = -1  # no visdom display
        
        self.netG_teacher = create_model(opt)
        self.netG_teacher.setup(opt)
        self.netG_teacher.eval()

        
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
        self.real_B = self.real_A
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def mae_loss(input, target):
        abs_diff = torch.abs(input - target)
        return torch.mean(abs_diff)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):

        self.teacher_output = self.netG_teacher.netG(self.fake_B)
        self.teacher_pred, self.teacher_uncertainty = torch.split(self.teacher_output, [3, 1], dim=1)
        self.teacher_weight_map = torch.abs(self.teacher_uncertainty - self.teacher_uncertainty.max()) + 1
        self.weighted_loss_tensor = self.teacher_uncertainty * self.teacher_weight_map
        self.pseudo_label = self.teacher_pred * self.weighted_loss_tensor

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 * 100
        
        self.eps = torch.tensor(1e-04).to(self.device)
        self.illum_gt = torch.div(self.real_A, torch.max(self.real_B, self.eps))
        self.illum_pred = torch.div(self.real_A, torch.max(self.fake_B, self.eps))

        self.student_loss = self.mae_loss(self.fake_B, self.pseudo_label)
        self.loss_G_Ang = self.criterionAngular(self.illum_gt, self.illum_pred) * self.opt.lambda_Angular

        self.loss_G = self.student_loss

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
