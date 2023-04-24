# Import libraries
import numpy as np
import torch
from .base_model import BaseModel
from . import networks, losses
from .losses import PatchNCELoss
import util.util as util

class CUTModel(BaseModel):
    
    """
    
    This class initializes a GAN model described in the paper. This code is heavily influenced by the PyTorch implementation of CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 
    
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train = True):
        
        """  
        
        This function adds arguments and returns parser based on the train options. 
        
        Parameters:
        
            parser   - initialized parser, parser object;
            is_train - option for training bool. 
            
        Output:
        
            parser   - parsed with added arguments, parser object.
            
        """
        
        # Start adding arguments to the parser
        parser.add_argument('--CUT_mode', type = str, default = "CUT", choices = '(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default = 1.0, help = 'weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default = 1.0, help = 'weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type = util.str2bool, nargs = '?', const = True, default = False, help = 'use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type = str, default = '0,4,8,12,16', help = 'compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type = util.str2bool, nargs = '?', const = True, default = False,
                            help = '(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type = str, default = 'mlp_sample', choices = ['sample', 'reshape', 'mlp_sample'], help = 'how to downsample the feature map')
        parser.add_argument('--netF_nc', type = int, default = 256)
        parser.add_argument('--nce_T', type = float, default = 0.07, help = 'temperature for NCE loss')
        parser.add_argument('--num_patches', type = int, default = 256, help = 'number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type = util.str2bool, nargs = '?', const = True, default = False,
                            help = "Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut": parser.set_defaults(nce_idt = True, lambda_NCE = 1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt = False, lambda_NCE = 10.0, flip_equivariance = True, n_epochs = 150, n_epochs_decay = 50)
        else: raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Losses list
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        
        # Names to be displayed
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        # Model names list
        self.model_names = ['G', 'F', 'D'] if self.isTrain else ['G']

        # Initialize generator and discriminator models
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        # Discriminator during training
        if self.isTrain: 
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # Loss functions for training
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers: self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            # Generator optimizer
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))

            # Discriminator optimizer
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas = (opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        
        """
        
        Feature network (netF) is initialized based on the shape of the intermediate features of the backbone (encoder) 
        Therefore, the weights of the feature network are initialized at the first feedforward pass with some input images.
        
        Parameters:
        
            data - input data, tensor.
        
        """
        
        # Get batch size
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        
        # Process the input data
        self.set_input(data)
        
        # Get real images from domain A and B
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        
        # Get generated images
        self.forward()
        
        if self.opt.isTrain:
            # Backpropagation for discriminator
            self.compute_D_loss().backward()
            # Backpropagation for generator
            self.compute_G_loss().backward()
            
            if self.opt.lambda_NCE > 0.0:
                # Feature network optimizer
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        
        """
        
        This function updates trainable parameters based-on the backpropagation results.
        
        """
        
        # Feed forward
        self.forward()

        # Discriminator network update
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        
        # Discriminator loss
        self.loss_D = self.compute_D_loss()
        
        # Discriminator loss backpropagation
        self.loss_D.backward()
        
        # Discriminator optimizer step
        self.optimizer_D.step()

        # Generator network update
        self.set_requires_grad(self.netD, False) # turn off gradient computation for discriminator
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample': self.optimizer_F.zero_grad()
        
        # Generator loss
        self.loss_G = self.compute_G_loss()
        
        # Generator loss backpropagation
        self.loss_G.backward()
        
        # Generator optimizer step
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample': self.optimizer_F.step()

    def set_input(self, input):
        
        """
        
        This function gets input tensor and does pre-preprocessing steps required for training.
        
        Parameters:
        
            input - input data, dictionary.
            
        """
        
        AtoB = self.opt.direction == 'AtoB'
        
        # Get real images from domain A and move them to device
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        
        # Get real images from domain B and move them to device
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        
        # Get images paths
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        
        """
        
        This function does feed forward of the network.
        
        """
        
        # Concatenate real images from domain A and B
        self.real = torch.cat((self.real_A, self.real_B), dim = 0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        
        # Apply flip
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        # Generate fake images based on the real ones
        self.fake = self.netG(self.real)
        
        # Generate fake image for domain B
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt: self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        
        """
        
        This function computes loss value for the discriminator.
        
        """
        
        # Detach a generated image for domain B
        fake = self.fake_B.detach()
        
        # Input the generated image through the discriminator
        pred_fake = self.netD(fake)
        
        # Get discriminator loss value for the generated image
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        
        # Input a real image through the discriminator
        self.pred_real = self.netD(self.real_B)
        
        # Get discriminator loss value for the real image
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # Compute and return discriminator loss
        return (self.loss_D_fake + self.loss_D_real) * 0.5

    def compute_G_loss(self):
        
        """
        
        This function computes loss value for the generator.
        
        """
        
        # Get a fake image from domain B
        fake = self.fake_B
        # G(A) tries should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            # Generate a fake image
            pred_fake = self.netD(fake)
            # Compute loss for the generator
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else: self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0: self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else: self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else: loss_NCE_both = self.loss_NCE

        return self.loss_G_GAN + loss_NCE_both

    def calculate_NCE_loss(self, src, tgt):
        
        """
        
        This function computes NCE loss value .
        
        Parameters:
        
            src      - a source image, tensor;
            tgt      - a target image, tensor.
            
        Output:
        
            loss     - loss value, tensor float.
        
        """
        
        # Get number of layers
        n_layers = len(self.nce_layers)
        
        # Query generation
        feat_q = self.netG(tgt, self.nce_layers, encode_only = True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance: feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        # Key generation
        feat_k = self.netG(src, self.nce_layers, encode_only = True)
        
        # Get key features and indices from the feature network
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        
        # Get query features from the feature network
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
