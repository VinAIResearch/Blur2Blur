import torch
from .base_model import BaseModel
from . import networks
from util.psnr_ssim import calculate_ssim
from util.util import normalize
from .losses import *
import yaml
from copy import deepcopy
from models.explore.kernel_encoding.kernel_wizard import KernelWizard
import random
from util.util import tensor2im
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from models.pg_modules.discriminator import ProjectedDiscriminator


class Blur2BlurModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """        
        parser.set_defaults(norm='batch', netG='unet_256')
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_Perc', type=float, default=0.8, help='weight for Perc loss')
            parser.add_argument('--lambda_gp', type=float, default=0.0001, help='weight for GP loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # breakpoint()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_Perc', 'D_real', 'D_fake']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.visual_names = ['real_A', 'blur_known', 'sharp_known', 'fake_B_']
        else:  # during test time, only load G
            self.model_names = ['G']
            self.visual_names = ['real_A', 'fake_B_']
        self.opt = opt

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.upscale = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        if self.isTrain:  
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionPerc = VGGLoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            self.netD.to(self.device)
            self.netD = torch.nn.DataParallel(self.netD)

        
        with open("options/generate_blur/augmentation.yml", "r") as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)["KernelWizard"]
            model_path = opt["pretrained"]
        self.genblur = KernelWizard(opt)
        print("Loading KernelWizard...")
        self.genblur.eval()
        self.genblur.load_state_dict(torch.load(model_path))
        self.genblur = self.genblur.to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.sizeA = input['sizeA']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        if self.isTrain:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            
            self.blur_known = input['C'].to(self.device)
            self.sharp_known = input['D'].to(self.device)
            
            kernel_mean, kernel_sigma = self.genblur(self.sharp_known, self.blur_known)
            self.kernel_real = kernel_mean + kernel_sigma * torch.randn_like(kernel_mean)
            self.real_B = self.genblur.adaptKernel(self.real_B, self.kernel_real)
            
    def deblurring_step(self, x):
        nbatch = x.shape[0]
        chunk_size = 4
        outs = []
        with torch.no_grad():
            for idx in range(0, nbatch, chunk_size):
                pred = self.deblur(x[idx: idx + chunk_size])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
        return torch.cat(outs, dim=0).to(self.device)
            

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B_ = self.fake_B[2]
        
    def backward_D(self, iters):
        """Calculate GAN loss for the discriminator"""

        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = [x.detach() for x in self.fake_B]
        pred_fake = self.netD(fake_B)
        self.loss_D_fake = self.criterionGAN(iters, pred_fake, False, dis_update=True)

        # Real
        real_B0 = F.interpolate(self.real_B, scale_factor=0.25, mode='bilinear')
        real_B1 = F.interpolate(self.real_B, scale_factor=0.5, mode='bilinear')
        real_B = [real_B0, real_B1, self.real_B]
        pred_real = self.netD(real_B)

        self.loss_D_real = self.criterionGAN(0, pred_real, True, dis_update=True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D += cal_gradient_penalty(
            self.netD, real_B[2], fake_B[2], self.real_B.device,
            self.opt.lambda_gp
        )[0]

        self.loss_D.backward()

    def backward_G(self, iters):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(0, pred_fake, True, dis_update=False)

        # Second, G(A) = A
        real_A0 = F.interpolate(self.real_A, scale_factor=0.25, mode='bilinear')
        real_A1 = F.interpolate(self.real_A, scale_factor=0.5, mode='bilinear')
        perc1 = self.criterionPerc.forward(self.fake_B[0], real_A0)
        perc2 = self.criterionPerc.forward(self.fake_B[1], real_A1)
        perc3 = self.criterionPerc.forward(self.fake_B[2], self.real_A) 
        self.loss_G_Perc = (perc1 + perc2 + perc3) * self.opt.lambda_Perc

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_Perc 
        self.loss_G.backward()

    def optimize_parameters(self, iters):
        self.forward()                   # compute fake images: G(A)

        # update D_kernel
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()             # set D's gradients to zero
        self.backward_D(iters)                   # calculate gradients for D
        self.optimizer_D.step()                  # update D's weights

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()              # set G's gradients to zero
        self.backward_G(iters)                    # calculate graidents for G
        self.optimizer_G.step()                   # update G's weights
