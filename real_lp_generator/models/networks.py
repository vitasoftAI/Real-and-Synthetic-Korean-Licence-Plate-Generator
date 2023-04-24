# Import libraries
import torch, timm, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np
from matplotlib import pyplot as plt
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator

##############################################################################################################################################################################
################################################################### Helper Functions #########################################################################################
##############################################################################################################################################################################

def pp(var_name, var, shape = False):
    
    """
    
    This function gets variable name, variable, and shape option and prints predefined variable with or without its shape.
    
    Parameters:
        
        var_name - variable name to be printed, str;
        var      - variable to be printed, var;
        shape    - shape option; prints shape of the variable if True else prints variable, bool.
    
    """
    
    if shape: print(f"{var_name} -> {var.shape}\n")        
    else: print(f"{var_name} -> {var}\n")

def get_filter(filt_size = 3):
    
    """
    
     This function gets filter size and returns tensor filter based on the given variable.
     
     Parameter:
         
         filt_size - filter size, int.
         
     Output:
     
        filt - filter, tensor.
     
    """

    # Initialize filter array
    if   (filt_size == 1): arr = np.array([1., ])
    elif (filt_size == 2): arr = np.array([1., 1.])
    elif (filt_size == 3): arr = np.array([1., 2., 1.])
    elif (filt_size == 4): arr = np.array([1., 3., 3., 1.])
    elif (filt_size == 5): arr = np.array([1., 4., 6., 4., 1.])
    elif (filt_size == 6): arr = np.array([1., 5., 10., 10., 5., 1.])
    elif (filt_size == 7): arr = np.array([1., 6., 15., 20., 15., 6., 1.])

    # Get filter
    filt = torch.Tensor(arr[:, None] * arr[None, :])
    
    return filt / torch.sum(filt)

def get_pad_layer(pad_type):
    
    """
    
    This function gets padding type as input and returns PadLayer.
    
    Parameters:
        
        pad_type - type of padding, str.
       
    Output:
    
        PadLayer - padding layer, torch.nn function.
    
    """
    
    assert pad_type in ['refl', 'reflect', 'repl', 'replicate', 'zero'], "Please choose a proper padding type."
    
    if (pad_type in ['refl', 'reflect']): PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']): PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'): PadLayer = nn.ZeroPad2d
    else: print(f'Pad type {pad_type} is not recognized!')
    
    return PadLayer

def get_norm_layer(norm_type = 'instance'):
        
    """
    
    This function gets normalization type and return a normalization layer.
    
    Parameters:
    
        norm_type - the name of the normalization layer: batch | instance | none, str;
    
    Output:
    
        norm_layer - a torch.nn Normalization layer.
        
    """
    
    assert norm_type in ['batch', 'instance', 'none'], "Please choose a proper norm type."
    
    if norm_type == 'batch': norm_layer = functools.partial(nn.BatchNorm2d, affine = True, track_running_stats = True)
    elif norm_type == 'instance': norm_layer = functools.partial(nn.InstanceNorm2d, affine = False, track_running_stats = False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    
    return norm_layer

def get_scheduler(optimizer, opt):
    
    
    """
    
    This function gets optimizer and options and returns a learning rate scheduler.
    
    Parameters:
    
        optimizer - the optimizer to update trainable parameters of the model.
        opt       - stores all the experiment flags; needs to be a subclass of BaseOptions．　

    Output:
        
        scheduler - learning rate scheduler.

    """
    
    assert opt.lr_policy in ['linear', 'step', 'plateau', 'cosine'], "Please choose a proper learning rate scheduler type."
    
    # Linear LR scheduler
    if opt.lr_policy == 'linear':
        
        def lambda_rule(epoch): return 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_rule)

    # Step LR scheduler
    elif opt.lr_policy == 'step': scheduler = lr_scheduler.StepLR(optimizer, step_size = opt.lr_decay_iters, gamma = 0.1)
    
    # Plateau LR scheduler
    elif opt.lr_policy == 'plateau': scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, threshold = 0.01, patience = 5)
    
    # Cosine LR scheduler
    elif opt.lr_policy == 'cosine':  scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = opt.n_epochs, eta_min = 0)
        
    return scheduler


def init_weights(model, init_type = 'normal', init_gain = 0.02, debug = False):
    
    """
    
    This function initializes network weights based on the given initialization type.
    
    Parameters:
    
        model - model to be initialized, model;
        init_type - the name of an initialization method;
        init_gain - scaling factor for normal, xavier and orthogonal, float.
        
    """
    
    assert init_type in ['normal', 'xavier', 'kaiming', 'orthogonal'], "Please choose a proper initialization type."
    
    def init_func(m):
        
        """
        
        This function initializes weigths.
        
        Parameter:
        
            m - model to get class name.
        
        """
        
        # Get classname
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug: print(classname)
            if init_type == 'normal': init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier': init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming': init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal': init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias.data, 0.0)
        
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # Apply the initialization function
    model.apply(init_func)  

def init_net(model, init_type = 'normal', init_gain = 0.02, gpu_ids = [], debug = False, initialize_weights = True):
    
    """
    
    This function initializes a cpu or gpu device model and initializing the model weights.
    
    Parameters:
    
        model     - the model to be initialized;
        init_type - the name of an initialization method, str;
        gain      - scaling factor for normal, xavier and orthogonal, float.
        gpu_ids   - which GPUs the network runs on, list.
    
    Output:
    
        an initialized model. 
    
    """
    
    if len(gpu_ids) > 0:
        
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights: init_weights(model, init_type, init_gain=init_gain, debug=debug)
    
    return model

def cal_gradient_penalty(netD, real_data, fake_data, device, type = 'mixed', constant = 1.0, lambda_gp = 10.0):
    
    """
    
    This function calculates the gradient penalty loss, used in WGAN-GP paper. 
    
    Parameters:
    
        netD        - discriminator network, model;
        real_data   - real images, tensor;
        fake_data   - generated images from the generator, tensor;
        device      - gpu or cpu device, str;
        type        - if we mix real and fake data or not [real | fake | mixed], str;
        constant    - the constant used in formula ( | |gradient||_2 - constant)^2, float;
        lambda_gp   - weight for this loss, float.
    
    Output:
    
        Gradient penalty loss.
    
    """
    
    assert type in ['real', 'fake', 'mixed'], "Please choose a proper type for gradient penalty."
    
    if lambda_gp > 0.0:
        
        # either use real images, fake images, or a linear interpolation of two.
        if type == 'real': interpolatesv = real_data
        elif type == 'fake': interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device = device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        
        return gradient_penalty, gradients
    
    else: return 0.0, None

def define_G(input_nc, output_nc, ngf, netG, norm = 'batch', use_dropout = False, init_type = 'normal',
             init_gain = 0.02, no_antialias = False, no_antialias_up = False, gpu_ids = [], opt = None):
    
    """
    
    This function creates a generator.
    
    Parameters:
    
        input_nc     - the number of channels in input images, int;
        output_nc    - the number of channels in output images, int;
        ngf          - the number of filters in the last conv layer, int;
        netG         - the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128, str;
        norm         - the name of normalization layers used in the network: batch | instance | none, str;
        use_dropout  - dropout option for the layers, bool;
        init_type    - the name of our initialization method; str;
        init_gain    - scaling factor for normal, xavier and orthogonal, float;
        gpu_ids      - gpu device name, list.
        
    Output:
    
        Generator.
    
    """
    
    assert netG in ['resnet_9blocks', 'unet_256', 'smallstylegan2', 'resnet_6blocks', 'resnet_4blocks', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], "Please choose a proper name for generator."
    
    net = None
    norm_layer = get_norm_layer(norm_type = norm)

    if   netG == 'resnet_9blocks': net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, no_antialias = no_antialias, no_antialias_up = no_antialias_up, n_blocks = 9, opt = opt)
    elif netG == 'resnet_6blocks': net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, no_antialias = no_antialias, no_antialias_up = no_antialias_up, n_blocks = 6, opt = opt)
    elif netG == 'resnet_4blocks': net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer = norm_layer, use_dropout = use_dropout, no_antialias = no_antialias, no_antialias_up = no_antialias_up, n_blocks = 4, opt = opt)
    elif netG == 'unet_128': net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer = norm_layer, use_dropout = use_dropout)
    elif netG == 'unet_256': net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer = norm_layer, use_dropout = use_dropout)
    elif netG == 'stylegan2': net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout = use_dropout, opt = opt)
    elif netG == 'smallstylegan2': net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout = use_dropout, n_blocks = 2, opt = opt)
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs = 2, n_res = n_blocks - 4, ngf = ngf, norm = 'inst', nl_layer = 'relu')
    
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))

def define_F(input_nc, netF, norm = 'batch', use_dropout = False, init_type = 'normal', init_gain = 0.02, no_antialias = False, gpu_ids = [], opt = None):
    
    """
    
    This function creates F network.
    
    Parameters:
    
        input_nc     - the number of channels in input images, int;
        netF         - F network name, str; 
        norm         - the name of normalization layers used in the network: batch | instance | none, str;
        use_dropout  - dropout option for the layers, bool;
        init_type    - the name of our initialization method; str;
        init_gain    - scaling factor for normal, xavier and orthogonal, float;
        gpu_ids      - gpu device name, list.
        
    Output:
    
        F Network.
    
    """
    
    assert netF in ['global_pool', 'reshape', 'sample', 'mlp_sample', 'strided_conv'], "Please choose a proper name for F network."
    
    if   netF == 'global_pool': net = PoolingF()
    elif netF == 'reshape': net = ReshapeF()
    elif netF == 'sample': net = PatchSampleF(use_mlp = False, init_type = init_type, init_gain = init_gain, gpu_ids = gpu_ids, nc = opt.netF_nc)
    elif netF == 'mlp_sample': net = PatchSampleF(use_mlp = True, init_type = init_type, init_gain = init_gain, gpu_ids = gpu_ids, nc = opt.netF_nc)
    elif netF == 'strided_conv': net = StridedConvF(init_type = init_type, init_gain = init_gain, gpu_ids = gpu_ids)

    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D = 3, norm = 'batch', init_type = 'normal', init_gain = 0.02, no_antialias = False, gpu_ids = [], opt = None):
    
    """
    
    This function creates a discriminator.
    
    Parameters:
    
        input_nc     - the number of channels in input images, int;
        ndf          - the number of filters in the first conv layer, int;
        netD         - the architecture's name: basic | n_layers | pixel, str;
        norm         - the type of normalization layers used in the model, str;
        init_type    - the name of our initialization method; str;
        init_gain    - scaling factor for normal, xavier and orthogonal, float;
        gpu_ids      - gpu device name, list.
        
    Output:
    
        Discriminator.
    
    """

    assert netD in ['basic', 'n_layers', 'pixel', 'stylegan2', 'strided_conv'], "Please choose a proper name for the discriminator network."
    disc = None
    norm_layer = get_norm_layer(norm_type = norm)

    # Default PatchGAN classifier
    if netD == 'basic':       disc = NLayerDiscriminator(input_nc, ndf, n_layers = 2, norm_layer = norm_layer, no_antialias = no_antialias)
    # Layer-based
    elif netD == 'n_layers':  disc = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer = norm_layer, no_antialias = no_antialias)
    # Pixel-based
    elif netD == 'pixel':     disc = PixelDiscriminator(input_nc, ndf, norm_layer = norm_layer)
    # StyleGAN
    elif 'stylegan2' in netD: disc = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias = no_antialias, opt = opt)
    
    return init_net(disc, init_type, init_gain, gpu_ids, initialize_weights = ('stylegan2' not in netD))

##############################################################################################################################################################################
######################################################################## Classes #############################################################################################
##############################################################################################################################################################################

class Identity(nn.Module):
    def forward(self, x): return x

class WIBReLU(nn.Module):
    
    
    """
    
    This class gets activations and compute their mean value and subtracts the mean value from the activations.
    This is implementation of the WiB-ReLU from https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.6143.
    
    """

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(WIBReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor: return F.relu(input, inplace=self.inplace) - torch.mean(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        
        return inplace_str

class Downsample(nn.Module):
    
    """
    
    This class downsamples an input tensor volume.
    
    Parameters:
    
        channels    - channels of the convolutioon filter, int;
        pad_type    - padding type, str;
        filt_size   - size of the convolution filter, int;
        stride      - a stride for the convolution filter, int;
        pad_off     - padding off, int.
        
    Output:
    
        downsampled tensor volume.
    
    """
    
    def __init__(self, channels, pad_type = 'reflect', filt_size = 3, stride = 2, pad_off = 0):
        
        super(Downsample, self).__init__()
        
        # Initialize class arguments
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # Get filters
        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        
        # Get padding layer
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        
        """
        
        This function gets input tensor volume and does downsampling.
        
        Parameter:
        
            inp - input image, tensor.
            
        Output:
    
            downsampled tensor volume.
        
        """
        
        if (self.filt_size == 1):
            
            if (self.pad_off == 0): return inp[:, :, ::self.stride, ::self.stride]
            else: return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        
        else: return F.conv2d(self.pad(inp), self.filt, stride = self.stride, groups = inp.shape[1])

class Upsample2(nn.Module):
    
    """
    
    This class upsamples an input tensor volume.
    
    Parameters:
    
        scale_factor - a factor for upsampling, int;
        mode         - a mode for upsampling, str.
        
    Output:
    
        upsampled tensor volume.
        
    """
    
    def __init__(self, scale_factor, mode = 'bilinear'):
        
        super().__init__()
        self.factor, self.mode = scale_factor, mode

    def forward(self, inp):
        
        """
        
        This function gets input tensor volume and does upsampling.
        
        Parameter:
        
            inp - input tensor volume, tensor.
        
        Output:
    
            upsampled tensor volume, tensor.
            
        """
        
        return torch.nn.functional.interpolate(inp, scale_factor = self.factor, mode = self.mode)

class Normalize(nn.Module):
    
    """
    
    This class normalizes an input tensor image.
    
    Parameter:
    
        x - an input image, tensor;
        
    Output:

        a normalized output image, tensor;
        
    """

    def __init__(self, power = 2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):

        norm = x.pow(self.power).sum(1, keepdim = True).pow(1. / self.power)
        
        return x.div(norm + 1e-7)

class PatchSampleF(nn.Module):
    
    """
    
    This class creates patches and return features along with the corresponding positions.
    
    Parameters:
    
        use_mlp      - option to use MLP or not, bool;
        init_type    - initializer type, str;
        init_gain    - weight for initialization, float;
        nc           - number of hidden layer neurons, int;
        gpu_ids      - gpu device ids, list.
        
    Outputs:
    
        return_feats - features to return, list; 
        return_ids   - indices of the features, list.
    
    """
    
    def __init__(self, use_mlp = False, init_type = 'normal', init_gain = 0.02, nc = 256, gpu_ids = []):

        super(PatchSampleF, self).__init__()
        self.l2norm    = Normalize(2)
        self.use_mlp   = use_mlp
        self.nc        = nc  
        self.mlp_init  = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids   = gpu_ids

    def create_mlp(self, feats):
        
        """
        
        This function gets features and creates multilayer perceptron.
        
        Parameter:
        
            feats - features, list.
            
        Output:
        
            MLP network, model.
        
        """
        
        # Go through the features list
        for mlp_id, feat in enumerate(feats):
            
            # Get input channels
            input_nc = feat.shape[1]
            
            # Initialize MLP
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            
            # Move to GPU
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        
        # Initialize the network
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches = 64, patch_ids = None):
        
        """
        
        This function gets features, number of patches and patch indices and returns features and their corresponding indices.
        
        Parameters:
        
            feats       - features, list;
            num_patches - number of patches, int;
            patch_ids   - patch indices, dict.
            
        Output:
        
            MLP network, model.
        
        """
        
        # Initialize lists
        return_ids, return_feats = [], []
        
        # Create MLP
        if self.use_mlp and not self.mlp_init: self.create_mlp(feats)
            
        # Go through the features list    
        for feat_id, feat in enumerate(feats):
            
            # if feat.shape[1] == 3:
                # plt.imshow((feat[0]).detach().cpu().permute(1,2,0).numpy().astype(np.uint8))
                # plt.savefig("sample.png")
            
            # Get batch, image height, and image width
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            
            # Reshape the feature map and flatten
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                
                # Get the patch id
                if patch_ids is not None: patch_id = patch_ids[feat_id]
                else:
                    # Get the patch id
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype = torch.long, device = feat.device)
                
                # Get the sample
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1) 
            else:
                # Get the sample
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                # Get the sample from MLP
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
            
        return return_feats, return_ids

class LinearBlock(nn.Module):
    
    """
    
    This class creates a fully connected layer and passes the input through it.
    
    Parameters:
    
        input_dim  - input dimension of the fully connected layer;
        output_dim - output dimension of the fully connected layer;
        norm       - normalization type, str;
        activation - activation function name, str.
    
    """
    
    
    def __init__(self, input_dim, output_dim, norm = 'none', activation = 'relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        
        # Initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias = use_bias)

        # Make sure the normalization and the activation function is one of the pre-defined normalizations, activation functions
        assert norm in ['batch', 'inst', 'ln', 'none'], "Please choose a proper normalization."
        assert activation in ['relu', 'lrelu', 'prelu', 'selu', 'tanh', 'wib', 'none'], "Please choose a proper activation function."
        
        # Initialize normalization
        norm_dim = output_dim
        if norm   == 'batch':self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst': self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':   self.norm = LayerNorm(norm_dim)
        elif norm == 'none': self.norm = None

        # Initialize activation function
        if activation == 'relu':    self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu': self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu': self.activation = nn.PReLU()
        elif activation == 'selu':  self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':  self.activation = nn.Tanh()
        elif activation == 'wib':  self.activation = WIBReLU()
        elif activation == 'none':  self.activation = None

    def forward(self, inp):
        
        """
        
        This function gets input volume, passes it through MLP layer, applies normalization and activation (if defined) and returns the output volume.
        
        Parameters:
        
            inp - input volume, tensor.
            
        Output:
        
            out - output volume, tensor.
        
        """
        
        # MLP
        out = self.fc(inp)
        # Normalization
        if self.norm:       out = self.norm(out)
        # Activation
        if self.activation: out = self.activation(out)
        
        return out

class ResnetGenerator(nn.Module):
    
    
    """
    
    This class creates Resnet-based generator that consists of Resnet blocks between a few downsampling and upsampling operations.
    The code is adapted from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style).
    
    Parameters:
    
        input_nc      - the number of channels in input volumes, int;
        output_nc     - the number of channels in output volumes, int;
        ngf           - the number of filters in the last convolution layer, int;
        norm_layer    - normalization layer name, torch class;
        use_dropout   - option to use dropout layers, bool;
        n_blocks      - the number of ResNet blocks, int;
        padding_type  - padding layer type name, str.
        
    Output:
    
        ResNet-based Generator model.
    
    """

    def __init__(self, input_nc, output_nc, ngf = 64, norm_layer = nn.BatchNorm2d, use_dropout = False, n_blocks = 6, padding_type = 'reflect', no_antialias = False, no_antialias_up = False, opt = None):
        
        # At least one ResNet block must be formulated
        assert(n_blocks >= 0), "At least one ResNet block must be initialized"
        
        super(ResnetGenerator, self).__init__()
        
        # Get options
        self.opt = opt
        # Get normalization layer
        if type(norm_layer) == functools.partial: use_bias = norm_layer.func == nn.InstanceNorm2d
        # Get use bias flag
        else: use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf // 2, kernel_size = 5, padding = 0, bias = use_bias), WIBReLU(True), nn.Conv2d(ngf // 2, ngf, kernel_size = 3, padding = 0, bias = use_bias), WIBReLU(True)]

        # Initialize depth for downsampling
        n_downsampling = 2
        
        # Create a model for downsampling 
        for i in range(n_downsampling):
            
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, 
                                stride=1, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True),
                      Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        # Create a model based on the number of ResNet blocks
        for i in range(n_blocks):       

            model += [ResnetBlock(ngf * mult, padding_type = padding_type, 
                                  norm_layer=norm_layer, use_dropout = use_dropout,
                                  use_bias = use_bias)]

        # Create a model for upsampling
        for i in range(n_downsampling):  
            mult = 2 ** (n_downsampling - i)
            model += [Upsample2(2),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                kernel_size = 3, stride = 1,
                                padding = 1, bias = use_bias),
                      norm_layer(int(ngf * mult / 2)), WIBReLU(True)]
        
        # Add padding layer
        model += [nn.ReflectionPad2d(3)]
        # Add final conv layer
        model += [nn.Conv2d(ngf, ngf // 2, kernel_size = 5, padding = 0, bias = use_bias), WIBReLU(True),
                   nn.Conv2d(ngf // 2, output_nc, kernel_size = 3, padding = 0, bias = use_bias)]
        # Add activation function
        model += [nn.Tanh()]
        
        # Formulate a final model
        self.model = nn.Sequential(*model)
    
    def forward(self, input, layers = [], encode_only = False):
        
        """
        
        This function gets input volume, layers, and encoding option; passes the input volume through the ResNet-based Generator model
        and outputs a generated image.
        
        Parameters:
        
            input       - input volume, tensor;
            layers      - pre-defined layers, list;
            encode_only - an option for encoding, bool.
            
        Output:
        
            fake       - a generated fake image, tensor.
        
        """
        
        if -1 in layers: layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    
                    # Return intermediate features only
                    return feats  

            # Return both output and intermediate features
            return feat, feats  
        
        # Standard feed forward network
        else: return self.model(input)

class ResnetBlock(nn.Module):
    
    """
    
    This class creates a Resnet block.
    
    Parameters:
    
        dim          - the number of channels in the convolution layer, int;
        padding_type - a padding layer type, str;
        norm_layer   - a normalization layer;
        use_dropout  - an option to use dropout, bool;
        use_bias     - an option to use bias in the convolution layers, bool.
        
    Output:
    
        a ResNet block, torch Sequential model.
    
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        
        """

        This function builds a convolution block.

        Parameters:

         dim          - the number of channels in the convolution layer, int;
         padding_type - a padding layer type, str;
         norm_layer   - a normalization layer;
         use_dropout  - an option to use dropout, bool;
         use_bias     - an option to use bias in the convolution layers, bool.

        Output:

         a ResNet block, torch Sequential model.

        """
        
        assert padding_type in ['reflect', 'replicate', 'zero'], "Please choose a proper padding type."
        
        # Initialize a list and padding variable
        conv_block, p = [], 0
        
        # Add padding layer
        if padding_type == 'reflect': conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': p = 1
        
        # Add conv blocks
        conv_block += [nn.Conv2d(dim, dim, kernel_size = 3, padding = p, bias = use_bias), nn.ReLU(True)]
        
        # Add dropout layer
        if use_dropout: conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, input):
        
        """
        
        This function gets input volume and passes it through the pre-defined residual block.
        
        Parameter:
        
            input       - input volume, tensor;
            
        Output:
        
            out        - output tensor after the residual block, tensor.
        
        """
        
        # Residual connection
        return input + self.conv_block(input)  

class NLayerDiscriminator(nn.Module):
    
    """
    
    This class creates a discriminator network.
    
    Parameters:
    
        input_nc     - input image channels, int;
        ndf          - number of channels in the first convolution layer, int;
        n_layers     - number of layers in the discriminator, int;
        norm_layer   - a normalization layer, torch method;
        no_antialias - an option for antialias, bool.
        
    Output:
    
        a discriminator model, torch Sequential model.
    
    """
    
    def __init__(self, input_nc, ndf = 64, n_layers = 3, norm_layer = nn.BatchNorm2d, no_antialias = False):
        
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ks, padw = 3, 1
        # 3, 64, 3, 1, 1 -> shape=(128, 128); channels=64
        sequence = [nn.Conv2d(input_nc, ndf // 2, kernel_size = ks, stride = 1, padding = padw), 
                    WIBReLU(True), nn.Conv2d(ndf // 2, ndf, ks, stride = 2, padding = padw), WIBReLU(True)] 
        
        # Go through the number of layers and gradually increase the number of filters
        for n in range(1, n_layers): 
            
            # Track the number of filters in the output of the previous convolution operation
            prev_channels = ndf * n if n == 1 else ndf * (n*2)
            
            sequence += [
                        nn.Conv2d(prev_channels, prev_channels // 4, kernel_size = 1, stride = 1, padding = 0, bias = use_bias), WIBReLU(True),
                        nn.Conv2d(prev_channels // 4, prev_channels * 2, kernel_size = ks, stride = 1, padding = padw, bias = use_bias), WIBReLU(True),
                        nn.Conv2d(prev_channels * 2, prev_channels // 2, kernel_size = 1, stride = 1, padding = 0, bias = use_bias), WIBReLU(True),
                        nn.Conv2d(prev_channels // 2, prev_channels * 4, kernel_size = ks, stride = 2, padding = padw, bias = use_bias), WIBReLU(True)
                        ]
            
        sequence += [
                        nn.Conv2d(prev_channels * 4, prev_channels // 2, kernel_size = 1, stride = 1, padding = 0, bias = use_bias), WIBReLU(True),
                        nn.Conv2d(prev_channels // 2, prev_channels * 8, kernel_size = ks, stride = 1, padding = padw, bias = use_bias), WIBReLU(True), 
                        nn.Conv2d(prev_channels * 8, prev_channels // 2, kernel_size = 1, stride = 1, padding = 0, bias = use_bias), WIBReLU(True),
                        nn.Conv2d(prev_channels // 2, 1, kernel_size = ks, stride = 1, padding = padw)
                    ] 
        # Create a final model
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        
        """
        
        This function gets input volume and passes it through the pre-defined discriminator model.
        
        Parameter:
        
            input       - input volume, tensor;
            
        Output:
        
            out        - output tensor after the residual block, tensor.
        
        """

        return self.model(input)
