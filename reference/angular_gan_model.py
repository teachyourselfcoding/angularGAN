from os import O_RDONLY
from pyexpat import model
import torch
import torchvision.transforms as transforms
from util.util import tensor2im
from util.util import save_image
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import cutmix
from . import angular_loss
from . import similarity
from . import net_canny
from math import pi
import random
torch.cuda.empty_cache()


class AngularGANModel(BaseModel):
    def name(self):
        return 'AngularGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_attn')

        parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--lambda_Angular', type=float, default=1.0, help='influence of angular loss')
        parser.add_argument('--lambda_surf_sim', type=float, default=0, help='influence of similarity loss')
        parser.add_argument('--lambda_illum_sim', type=float, default=0, help='influence of similarity loss')
        parser.add_argument('--lambda_contrast', type=float, default=0, help='influence of contrastive loss')
        parser.add_argument('--lambda_discriminator', type=float, default=0, help='influence of discriminator loss')
        parser.add_argument('--lambda_aux', type=float, default=0, help='influence of auxiliary task loss')
        parser.add_argument('--lambda_aux_illum', type=float, default=0, help='influence of achromatic pixel detection')
        parser.add_argument('--lambda_aux_edge', type=float, default=0, help='influence of edge detection')
        parser.add_argument('--nce_T', type=float, default=1, help='temperature of NCE loss calculation') # 0.87
        parser.add_argument('--num_patches', type=float, default=64, help='number of patches on features for contrastive learning')
        parser.add_argument('--use_uncertainty', action='store_true', help='use uncertainty')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.eps = torch.tensor(1e-4).to(self.device)
        self.idx = 0
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_L1', 'G_Ang']
        if self.isTrain and self.opt.lambda_contrast > 0:
            self.loss_names += ['contrast']
        if self.isTrain and self.opt.lambda_aux > 0:
            self.loss_names += ['G_aux']
        if self.isTrain and self.opt.lambda_surf_sim > 0:
            self.loss_names += ['surf_sim']
        if self.isTrain and self.opt.lambda_illum_sim > 0:
            self.loss_names += ['illum_sim']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B','est_chrom','gt_chrom','zero_mask','err_map']
        if self.opt.use_uncertainty:
            self.visual_names += ['var_map']
        if self.isTrain and self.opt.lambda_aux > 0:
                self.visual_names += ['aux_map_illum', 'aux_map_edge']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        if self.isTrain:
            if opt.lambda_aux>0:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, aux_num=2)
            else:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, aux_num=0)
        else:
            if opt.lambda_aux>0:
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, aux_num=2)
            else:    
                self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                                opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, aux_num=0)
        
        self.criterionAngular   = angular_loss.angular_loss()
        self.criterionSim     = similarity.IllumCorrelativeLoss()
        self.temp_real_A = None
        #self.temp_real_B = None
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.use_uncertainty:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc -1, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Canny Edge Detection
            self.canny = net_canny.CannyNet(threshold=3.0, use_cuda=True).to(self.device)
            self.canny.cuda()
            self.canny.eval()

            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionNCE  = networks.PatchNCELoss(self.opt).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionSRC = networks.SRCLoss(dim=1)
            
            self.colorjitter = transforms.ColorJitter(hue=0.5)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        else:
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(),lr=0)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #self.real_A = (self.real_A+1.0) /2.0
        #self.real_B = (self.real_B+1.0) /2.0
        #self.real_A = torch.max(self.real_A, self.eps)
        #self.real_B = torch.max(self.real_B, self.eps)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.zero_mask = input['zero_mask'].to(self.device)

        #if self.isTrain and self.opt.lambda_aux > 0.0:
        #   self.seg = input['seg'].to(self.device)

    def forward(self):
        if self.opt.lambda_aux > 0.0:
            self.fake_B, self.aux_B_1, self.aux_B_2 = self.netG(self.real_A)
        else:
            # self.fake_B,_,_ = self.netG(self.real_A)
            self.fake_B = self.netG(self.real_A)
        # print(self.fake_B)
        self.fake_B = torch.clamp(self.fake_B,-1.0, 1.0)

        self.illum_pred = torch.div((self.real_A+1.0)/2.0, torch.max((self.fake_B+1.0)/2.0, self.eps))
        self.illum_gt   = torch.div((self.real_A+1.0)/2.0, torch.max((self.real_B+1.0)/2.0, self.eps))

        # get chromacity
        self.gt_chrom   = torch.div((self.real_A+1.0)/2.0, torch.max((self.real_B+1.0)/2.0, self.eps))
        self.est_chrom  = torch.div((self.real_A+1.0)/2.0, torch.max((self.fake_B+1.0)/2.0, self.eps))
        #self.est_chrom = torch.div(self.real_A, torch.max(self.fake_B, self.eps))
        #self.gt_chrom   = torch.div(self.real_A, torch.max(self.real_B, self.eps))
        self.gt_sum       = torch.sum(self.gt_chrom,dim=1).view(-1,1,256,256)
        self.est_sum      = torch.sum(self.est_chrom,dim=1).view(-1,1,256,256)
        self.gt_chrom     = self.gt_chrom/self.gt_sum
        self.est_chrom    = self.est_chrom/self.est_sum
        
        # anglar loss map
        self.err = self.criterionAngular(self.illum_gt, self.illum_pred)
        self.err = self.err.reshape(-1,1,256,256)
        self.err_map = torch.cat((self.err,self.err,self.err),1)
        self.err_map = self.err_map/45.0 - 1.0
        self.err_map = self.err_map.reshape(-1,3,256,256)
        self.err_map = self.err_map*self.zero_mask
        if self.opt.use_uncertainty:
          self.var = self.fake_B_all[:, 3, :, :]
          self.var = torch.clamp(self.var, min=0.001)
          self.var_map = self.var.view([-1,1,256,256])
          self.var_map = self.var_map*2.0/self.var_map.max() - 1.0
        
        # contrastive part
        
        # MTL - AUX
        if self.isTrain and self.opt.lambda_aux > 0.0:
            # Achromatic Pixel Detection
            self.aux_map_illum = torch.cat((self.aux_B_1,self.aux_B_1,self.aux_B_1),1)
            estimate = torch.sum((self.real_B+1.0)/2.0 * self.aux_map_illum, dim=(2,3), keepdim=True)
            self.illum_achrom = torch.nn.functional.normalize(estimate + 1e-2 * torch.randn_like(estimate))
            
            # Edge Detection
            th = []
            for i in range(self.real_A.shape[0]):
              blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = self.canny(self.real_A[i:(i+1)])
              th.append(thresholded.detach())
            self.edge = torch.cat(tuple(th),dim=0)
            self.aux_map_edge = torch.cat((self.aux_B_2,self.aux_B_2,self.aux_B_2),1) * 2 - 1

            
            

    def backward_D(self, isOptimize=True):

        if self.opt.which_model_netD == 'unet':
            # Fake
            # stop backprop to the generator by detaching fake_B
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            pred_fake_global, pred_fake_pixelwise = self.netD(fake_AB.detach())
            self.loss_D_fake_global = self.criterionGAN(pred_fake_global, False)
            self.loss_D_fake_pixelwise = self.criterionGAN(pred_fake_pixelwise, False)
            self.loss_D_fake = self.loss_D_fake_global + self.loss_D_fake_pixelwise

            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real_global, pred_real_pixelwise = self.netD(real_AB)
            self.loss_D_real_global = self.criterionGAN(pred_real_global, True)
            self.loss_D_real_pixelwise = self.criterionGAN(pred_real_pixelwise, True)
            self.loss_D_real = self.loss_D_real_global + self.loss_D_real_pixelwise

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 
            self.loss_D.backward()

        else:
            # Fake
            # stop backprop to the generator by detaching fake_B
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B.detach()), 1))
            pred_fake = self.netD(fake_AB.detach())         
            self.loss_D_fake = self.criterionGAN(pred_fake, False)

            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB.detach())
            self.loss_D_real = self.criterionGAN(pred_real, True)

            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()

    def backward_G(self):
        # Discriminator loss
        if self.opt.which_model_netD =='unet':
            # mix cut only for batch size = 1
            # self.mix_map = cutmix.CutMix(self.opt.fineSize)
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake_global, pred_fake_pixelwise = self.netD(fake_AB)
            self.loss_G_GAN_global = self.criterionGAN(pred_fake_global, True)
            self.loss_G_GAN_pixelwise = self.criterionGAN(pred_fake_pixelwise, True)
            self.loss_G_GAN = (self.loss_G_GAN_global + self.loss_G_GAN_pixelwise) * self.opt.lambda_discriminator
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_discriminator  
                    
        # L1/MSE loss
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 * 100
        self.loss_G_L1 = self.criterionMSE(self.fake_B, self.real_B) * self.opt.lambda_L1 * 100

        # angular loss
        if self.opt.use_uncertainty:
            # self.loss_G_Ang = self.uncertaintyLoss(self.illum_gt, self.illum_pred, self.var) * self.opt.lambda_Angular
            self.loss_G_Ang = torch.mean(self.criterionAngular(self.illum_gt, self.illum_pred)) * self.opt.lambda_Angular
        else:
            self.loss_G_Ang = torch.mean(self.criterionAngular(self.illum_gt, self.illum_pred)) * self.opt.lambda_Angular
        # self-similarity loss
        self.norm_fake_B = torch.max((self.fake_B+1.0)/2.0, self.eps)
        self.norm_real_B = torch.max((self.real_B+1.0)/2.0, self.eps)
        self.loss_surf_sim  = self.criterionSim.loss(self.norm_fake_B,self.norm_real_B) * self.opt.lambda_surf_sim
        self.loss_illum_sim = 0

        self.loss_G = self.loss_G_Ang + self.loss_G_L1
        self.loss_G += self.loss_G_GAN
        self.loss_G += self.loss_surf_sim
        self.loss_G += self.loss_illum_sim

        #contrastive loss
        if self.opt.lambda_contrast > 0.0:
            self.loss_contrast = self.calculate_NCE_loss(self.real_A, self.fake_B)
            self.loss_G += self.loss_contrast
        
        #aux
        if self.opt.lambda_aux > 0.0:
                self.loss_G_aux = torch.mean(1-(torch.sum(self.illum_achrom, dim=1, keepdim=True)/(1e-4+torch.sqrt(3*(torch.sum(self.illum_achrom ** 2, dim=1, keepdim=True)))))) * self.opt.lambda_aux_illum
                self.loss_G_aux += self.criterionL1(self.edge, self.aux_B_2) * self.opt.lambda_aux_edge
                
                self.loss_G += self.loss_G_aux * self.opt.lambda_aux

        self.loss_G.backward()
    

    def calculate_NCE_loss(self, src, tgt):
        feat_t = self.netG(tgt, encoder_only=True) # target features
        feat_s = self.netG(src, encoder_only=True) # source features

        feat_s_, sample_ids = self.netG(feat_s, contrast_sample=True, num_patches=self.opt.num_patches, patch_id=None) # 64,512
        feat_t_, _          = self.netG(feat_t, contrast_sample=True, num_patches=self.opt.num_patches, patch_id=sample_ids) # 64,512

        loss = self.criterionNCE(feat_t_,feat_s_)* self.opt.lambda_contrast
        #loss = (self.criterionNCE(feat_t_,feat_s_) + self.criterionSRC(feat_t_,feat_s_))* self.opt.lambda_contrast

        return loss.mean()


    def optimize_parameters(self):
        self.forward()
        # update D
        # self.set_requires_grad(self.netD, True)
        # self.optimizer_D.zero_grad()
        # self.backward_D()
        # self.optimizer_D.step()

        # update G & MLP
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update temp_real_A
        self.temp_real_A = self.real_A.detach()
        #self.temp_real_B = self.real_B.detach()
    
    def test_time_training(self):
        self.forward()
        
        self.optimizer_G.zero_grad()

        # Backward G
        self.loss_G_aux = self.criterionL1(self.real_A, self.aux_B) * self.opt.lambda_aux
        #self.loss_G_aux = (1-(torch.sum(self.illum_achrom)/(1e-4+torch.sqrt(3*(torch.sum(torch.pow(self.illum_achrom,2))))))) * self.opt.lambda_aux
        self.loss_G_aux.backward()

        self.optimizer_G.step()

        with torch.no_grad():
            self.forward()