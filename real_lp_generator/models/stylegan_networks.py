# Import libraries
import torch, math, random
import numpy as np
from torch import nn
from torch.nn import functional as F

def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        :,
        max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up = 1, down = 1, pad = (0, 0)): return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

class PixelNorm(nn.Module):
    
    """
    
    This class computes pixel norm.
    
    Parameter:
    
        input       - input volume, tensor.
        
    Output:
    
        pixel_norm  - pixel norm value, tensor.
    
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim = 1, keepdim = True) + 1e-8)

def make_kernel(k):
    
    """
    
    This function makes kernel.
    
    Parameter:
    
        k - kernel array, array.
        
    Output:
    
        k - kernel tensor, tensor.
    
    """
    
    # Change to tensor
    k = torch.tensor(k, dtype = torch.float32)

    if len(k.shape) == 1: k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Upsample(nn.Module):
    
    """
    
    This class gets input tensor volume and increases its dimensions by a factor of 2.
    
    Parameters:
    
        kernel - size of the kernel;
        factor - a factor size to upsample, int.
        
    Output:
    
        output - upsampled output volume, tensor.
    
    """
    
    def __init__(self, kernel, factor = 2):
        super().__init__()

        self.factor = factor
        # Get kernel
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        # Padding
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input): return upfirdn2d(input, self.kernel, up = self.factor, down = 1, pad = self.pad)

class Downsample(nn.Module):
    
    """
    
    This class gets input tensor volume and decreases its dimensions by a factor of 2.
    
    Parameters:
    
        kernel - size of the kernel;
        factor - a factor size to downsample, int.
        
    Output:
    
        output - downsampled output volume, tensor.
    
    """
    
    def __init__(self, kernel, factor = 2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input): return upfirdn2d(input, self.kernel, up = 1, down = self.factor, pad = self.pad)

class Blur(nn.Module):
    
    """
    
    This class gets input tensor volume and does blur operation.
    
    Parameters:
    
        kernel          - size of the kernel;
        pad             - padding function;
        upsample_factor - a factor size to upsample, int.
        
    Output:
    
        output - blurred output volume, tensor.
    
    """
    
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1: kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input): return upfirdn2d(input, self.kernel, pad=self.pad)

class EqualConv2d(nn.Module):
    
    """
    
    This class conducts equal convolution operation. 
    
    Parameters:
    
        in_channel    - number of channels of an input volume, int;
        out_channel   - number of channels of an output volume, int;
        kernel_size   - kernel size of the convolution operation, int.
        
    Output:
    
        out           - output volume from the class, tensor.
    
    """
    
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        # Get weights and scale value
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = math.sqrt(1) / math.sqrt(in_channel * (kernel_size ** 2))

        # Get stride and padding values
        self.stride, self.padding = stride, padding

        # Initialize bias parameter
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input):
        # print("Before EqualConv2d: ", input.abs().mean())
        out = F.conv2d(input, self.weight * self.scale, bias = self.bias, stride = self.stride, padding = self.padding)
        # print("After EqualConv2d: ", out.abs().mean(), (self.weight * self.scale).abs().mean())
        return out

    def __repr__(self): return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},' f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')


class EqualLinear(nn.Module):
    
    """
    
    This class conducts equal linear matrix multiplication operation. 
    
    Parameters:
    
        in_dim    - number of dimensions of an input volume, int;
        out_dim   - number of dimensions of an output volume, int;
        
    Output:
    
        out       - output volume from the class, tensor.
    
    """
    
    def __init__(self, in_dim, out_dim, bias = True, bias_init = 0, lr_mul = 1, activation = None):
        super().__init__()

        # Get weight parameter
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        # Get bias parameter
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None

        # Get activation and multiply factor
        self.activation, self.lr_mul = activation, lr_mul

        # Get scale
        self.scale = (math.sqrt(1) / math.sqrt(in_dim)) * lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else: out = F.linear(input, self.weight * self.scale, bias = self.bias * self.lr_mul)
        return out

    def __repr__(self): return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class ScaledLeakyReLU(nn.Module):
    
    """
    
    This class performs leaky ReLU activate function with a pre-defined scale. 
    
    Parameters:
    
        negative_slope    - value for leaky ReLU's negative slope, float;
        
    Output:
    
        out               - output from the scaled leaky ReLU, tensor.
    
    """

    def __init__(self, negative_slope = 0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input): F.leaky_relu(input, negative_slope = self.negative_slope) * math.sqrt(2)

class ModulatedConv2d(nn.Module):
    
    """
    
    This class performs modulated convolution operation. 
    
    Parameters:
    
        in_channel        - number of channels for the input volume to the convolution operation, int;
        out_channel       - number of channels for the out volume from the convolution operation, int;
        kernel_size       - size of the filter of the convolution operation, int;
        style_dim         - style dimension, int;
        demodulate        - whether or not to use demoluated convolution, bool;
        upsample          - whether or not to use upsampling, bool;
        downsample        - whether or not to use downsampling, bool;
        blur_kernel       - kernel size for blurring, list -> int.
        
    Output:
    
        out               - output from the scaled leaky ReLU, tensor.
    
    """
    
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate = True,
        upsample = False,
        downsample = False,
        blur_kernel = [1, 3, 3, 1],
    ):
        super().__init__()

        self.eps, self.kernel_size,  self.in_channel, self.out_channel, self.upsample, self.downsample = 1e-8, kernel_size, in_channel, out_channel, upsample, downsample
        
        # Upsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad = (pad0, pad1), upsample_factor = factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = math.sqrt(1) / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        if style_dim is not None and style_dim > 0: self.modulation = EqualLinear(style_dim, in_channel, bias_init = 1)
        self.demodulate = demodulate

    def __repr__(self): return (f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, ' f'upsample = {self.upsample}, downsample = {self.downsample})')

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) if style is not None else torch.ones(batch, 1, in_channel, 1, 1).cuda()
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding = 0, stride = 2, groups = batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding = 0, stride = 2, groups = batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    
    """
    
    This class applies Noise to an input image.
    
    Parameters:
    
        image    - input image, tensor;
        noise    - noise level, float.
    
    """
    
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise = None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    
    """
    
    This class creates and returns constant input parameter.
    
    Parameters:
    
        channel   - channel size, int;
        size      - size of the input to be created, int.
        
    Output:
    
        out      - constant input, tensor.
    
    """
    
    def __init__(self, channel, size = 4):
        super().__init__()

        # Initialize input parameter
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        
        # Get batch size
        batch = input.shape[0]
        
        # Create constant to be returned
        return self.input.repeat(batch, 1, 1, 1)

class StyledConv(nn.Module):
    
    
    """
    
    This class creates styles and conducts styled convolution operation.
    
    Parameters:
    
        in_channel        - number of channels for the input volume to the convolution operation, int;
        out_channel       - number of channels for the out volume from the convolution operation, int;
        kernel_size       - size of the filter of the convolution operation, int;
        style_dim         - style dimension, int;
        demodulate        - whether or not to use demoluated convolution, bool;
        upsample          - whether or not to use upsampling, bool;
        downsample        - whether or not to use downsampling, bool;
        blur_kernel       - kernel size for blurring, list -> int;
        inject_noise      - whether or not to add noise, bool.
    
    """
    
    def __init__(self, in_channel, out_channel, kernel_size, style_dim = None, upsample = False, blur_kernel = [1, 3, 3, 1], demodulate = True, inject_noise = True):
        super().__init__()

        self.inject_noise = inject_noise
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample = upsample,
            blur_kernel = blur_kernel,
            demodulate = demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        if self.inject_noise:
            out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    
    
    """
    
    This class gets input channels and style dimension and changes image to RGB.
    
    Parameters:
    
        in_channel  - number of channels in input volume, int;
        style_dim   - style dimension, int;
        upsample    - whether or not to upsample, bool;
        blur_kernel - kernel size for blurring, list -> int.
        
    Output:
    
        out         - output volume from the class, tensor.
    
    """
    
    def __init__(self, in_channel, style_dim, upsample = True, blur_kernel = [1, 3, 3, 1]):
        super().__init__()

        # Upsample
        if upsample: self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip = None):
        
        """
        This function gets input volume and conducts feed forward of the class.
        
        Parameters:
        
            input    - input volume to the class, tensor;
            style    - type of style, str;
            skip     - whether or not to use skip connection, bool.
            
        Output:
        
            out      - output volume from the class, tensor.
        
        """
        
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            
            skip = self.upsample(skip)
            out = out + skip

        return out


class Generator(nn.Module):
    
    """
    
    This class gets several parameters and returns generated style image.
    
    Parameters:
    
        size                - size, int;
        style_dim           - style dimension, int;
        n_mlp               - number of mlps, int;
        channel_multiplier  - multiplier channels, int;
        blur_kernel         - kernel size for blurring, list -> int.
        lr_mlp              - mlp learning rate value, float.
        
    Output:
    
        image               - a generated style image, tensor.
    
    """
    
    def __init__(self, size, style_dim, n_mlp, channel_multiplier = 2, blur_kernel = [1, 3, 3, 1], lr_mlp = 0.01):
        super().__init__()
        self.size, self.style_dim = size, style_dim
        
        # Initialize layers
        layers = [PixelNorm()]

        for i in range(n_mlp): layers.append(  EqualLinear(style_dim, style_dim, lr_mul = lr_mlp, activation = "fused_lrelu")  )

        # Initialize styles model
        self.style = nn.Sequential(*layers)

        # Set number of channels
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv( self.channels[4], self.channels[4], 3, style_dim, blur_kernel = blur_kernel )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample = False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs, self.to_rgbs, self.upsamples, self.noises = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.Module()
        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(  StyledConv(in_channel, out_channel, 3, style_dim, upsample = True, blur_kernel = blur_kernel)   )
            self.convs.append(  StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel = blur_kernel)  )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        
        """
        
        This function creates noise to be applied to an image.
        
        """
        
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device = device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2): noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device = device))
        
        return noises

    def mean_latent(self, n_latent):
        
        """
        
        This function gets a parameter creates latent.
        
        Parameter:
        
            n_latent    - number of latents, int.
            
        Output:
        
            latent      - styled latent, tensor.
        
        """
        
        latent_in = torch.randn( n_latent, self.style_dim, device = self.input.input.device )
        latent = self.style(latent_in).mean(0, keepdim = True)

        return latent

    def get_latent(self, input): return self.style(input)

    def forward(
        self,
        styles,
        return_latents = False,
        inject_index = None,
        truncation = 1,
        truncation_latent = None,
        input_is_latent = False,
        noise = None,
        randomize_noise = True,
    ):
        if not input_is_latent: styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(  truncation_latent + truncation * (style - truncation_latent)  )
            styles = style_t

        if len(styles) < 2: inject_index = self.n_latent

            if len(styles[0].shape) < 3: latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else: latent = styles[0]

        else:
            if inject_index is None: inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(  self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs  ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents: return image, latent

        else: return image, None

class ConvLayer(nn.Sequential):
    
    """
    
    This class initializes convolution layer for style GAN. 
    
    Parameters:
        
        in_channel        - number of channels for the input volume to the convolution operation, int;
        out_channel       - number of channels for the out volume from the convolution operation, int;
        kernel_size       - size of the filter of the convolution operation, int;
        downsample        - whether or not to use downsampling, bool;
        blur_kernel       - kernel size for blurring, list -> int;
        bias              - whether or not to use bias, bool;
        activate          - whether or not to use activation function, bool.
    
    """
    
    def __init__(
        self,
        in_channel, out_channel,
        kernel_size, downsample = False,
        blur_kernel = [1, 3, 3, 1],
        bias = True, activate = True,
    ):
        
        # Initialize layers list
        layers = []

        if downsample:
            
            # Factor for donwsampling
            factor = 2
            
            # Padding values
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0, pad1 = (p + 1) // 2, p // 2

            layers.append(Blur(blur_kernel, pad = (pad0, pad1)))
            stride, self.padding = 2, 0

        layers.append( EqualConv2d( in_channel, out_channel, kernel_size, padding = self.padding, stride = stride, bias = bias and not activate  )  )

        if activate:
            if bias: layers.append(FusedLeakyReLU(out_channel))

            else: layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)

class ResBlock(nn.Module):
    
    """
    
    This class initializes residual block for style GAN.
    
    Parameters:
        
        in_channel        - number of channels for the input volume to the convolution operation, int;
        out_channel       - number of channels for the out volume from the convolution operation, int;
        downsample        - whether or not to use downsampling, bool;
        blur_kernel       - kernel size for blurring, list -> int;
        skip_gain         - gain value, float.
    
    """
    
    def __init__(self, in_channel, out_channel, blur_kernel = [1, 3, 3, 1], downsample = True, skip_gain = 1.0):
        super().__init__()

        self.skip_gain, self.conv1, self.conv2 = skip_gain, ConvLayer(in_channel, in_channel, 3), ConvLayer(in_channel, out_channel, 3, downsample = downsample, blur_kernel = blur_kernel)

        if in_channel != out_channel or downsample:
            self.skip = ConvLayer( in_channel, out_channel, 1, downsample = downsample, activate = False, bias = False )
        else: self.skip = nn.Identity()

    def forward(self, input):
        
        # Get output from conv layers
        out = self.conv2(self.conv1(input))
        
        # Get output from residual convolution
        skip = self.skip(input)
        
        # Return output of the residual block
        return (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)

class StyleGAN2Discriminator(nn.Module):
    
    """
    
    This class gets several parameters and implements discrimination phase of StyleGan.
    
    Parameters:
    
        input_nc        - number of channels in input volume, int;
        ndf             - number of channels in output volume, int;
        n_layers        - number of layers in the discriminator, int;
        no_antialias    - whether or not to use antialias, bool;
        size            - size of the network, int;
        opt             - options, parser object.
        
    """
    
    def __init__(self, input_nc, ndf = 64, n_layers = 3, no_antialias = False, size = None, opt = None):
        super().__init__()
        
        self.opt, self.stddev_group = opt, 16
        
        if size is None:
            size = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
            if "patch" in self.opt.netD and self.opt.D_patch_size is not None:
                size = 2 ** int(np.log2(self.opt.D_patch_size))

        # Initialize blur kernel
        blur_kernel = [1, 3, 3, 1]
        
        # Get channel multiplier
        channel_multiplier = ndf / 64
        
        # Initialize channels dictionary
        channels = {
            4: min(384, int(4096 * channel_multiplier)),
            8: min(384, int(2048 * channel_multiplier)),
            16: min(384, int(1024 * channel_multiplier)),
            32: min(384, int(512 * channel_multiplier)),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        # Get convolution layers list
        convs = [ConvLayer(3, channels[size], 1)]
        
        # Get log size and input channel
        log_size, in_channel = int(math.log(size, 2)), channels[size]

        if "smallpatch" in self.opt.netD: final_res_log2 = 4
        elif "patch" in self.opt.netD: final_res_log2 = 3
        else: final_res_log2 = 2

        for i in range(log_size, final_res_log2, -1):
            
            # Get output channels
            out_channel = channels[2 ** (i - 1)]
            
            # Add to the convolutions list
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            
            # Set input channel
            in_channel = out_channel

        # Initialize convolution sequential object
        self.convs = nn.Sequential(*convs)

        if False and "tile" in self.opt.netD: in_channel += 1
        
        # Set final convolution
        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        
        # Initialize final linear layer for patch case
        if "patch" in self.opt.netD: self.final_linear = ConvLayer(channels[4], 1, 3, bias = False, activate = False)
        
        # Initialize final linear layer for other case
        else: self.final_linear = nn.Sequential( EqualLinear(channels[4] * 4 * 4, channels[4], activation = "fused_lrelu"), EqualLinear(channels[4], 1) )

    def forward(self, input, get_minibatch_features = False):
        
        """
        
        This function gets several parameters and conducts feedforward of StyleGAN2Discriminator class.
        
        Parameters:
        
            input                  - input volume, tensor;
            get_minibatch_features - whether or not minibatch features, bool.
        
        Output:
        
            out                    - output tensor from the class, tensor.
        
        """
        
        if "patch" in self.opt.netD and self.opt.D_patch_size is not None:
            h, w = input.size(2), input.size(3)
            y = torch.randint(h - self.opt.D_patch_size, ())
            x = torch.randint(w - self.opt.D_patch_size, ())
            input = input[:, :, y:y + self.opt.D_patch_size, x:x + self.opt.D_patch_size]
        
        out = input
        
        # Go through convolution layers
        for i, conv in enumerate(self.convs): out = conv(out)
        
        # Get information from the convolution output
        batch, channel, height, width = out.shape

        if False and "tile" in self.opt.netD:
            group = min(batch, self.stddev_group)
            stddev = out.view(group, -1, 1, channel // 1, height, width)
            stddev = torch.sqrt(stddev.var(0, unbiased = False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdim = True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)
        
        # Get output from final convolution
        out = self.final_conv(out)

        # For patch case
        if "patch" not in self.opt.netD: out = out.view(batch, -1)
        
        # For other case
        out = self.final_linear(out)

        return out

class TileStyleGAN2Discriminator(StyleGAN2Discriminator):
    
    """
    
    This class gets an input and performs tile style GAN discriminator.
    
    Parameter:
    
        input    - an input volume, tensor.
    
    Output:
    
        out      - an output volume from the class, tensor.
    
    """
    
    def forward(self, input):
        
        # Get batch size, channels, and image dimensions from the input tensor
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        
        # Get patch size
        size = self.opt.D_patch_size
        
        # Get image dimensions to create patches
        Y, X = H // size, W // size
         
        # Create patches
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        
        return super().forward(input)

class StyleGAN2Encoder(nn.Module):
    
    """
    
    This class gets several parameters and performs encoder of the StyleGAN network.
    
    Parameters:
    
        input_nc        - number of channels in input volume, int;
        output_nc       - number of channels in output volume, int;
        ngf             - number of gan filters, int;
        use_dropout     - whether or not to use dropout, bool;
        n_blocks        - number of blocks, int;
        padding_type    - type of padding, str;
        no_antialias    - whether or not to use antialias, bool;
        opt             - options, parser object.  
    
    """
    
    def __init__(self, input_nc, output_nc, ngf = 64, use_dropout = False, n_blocks = 6, padding_type = "reflect", no_antialias = False, opt = None):
        super().__init__()
        assert opt is not None
        self.opt = opt
        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        blur_kernel = [1, 3, 3, 1]

        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
        convs = [nn.Identity(),
                 ConvLayer(3, channels[cur_res], 1)]

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res // 2]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, downsample=True))
            cur_res = cur_res // 2

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        self.convs = nn.Sequential(*convs)

    def forward(self, input, layers=[], get_features=False):
        feat = input
        feats = []
        if -1 in layers:
            layers.append(len(self.convs) - 1)
        for layer_id, layer in enumerate(self.convs):
            feat = layer(feat)
            # print(layer_id, " features ", feat.abs().mean())
            if layer_id in layers:
                feats.append(feat)

        if get_features:
            return feat, feats
        else:
            return feat


class StyleGAN2Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, opt=None):
        super().__init__()
        assert opt is not None
        self.opt = opt

        blur_kernel = [1, 3, 3, 1]

        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size))))) // (2 ** num_downsampling)
        convs = []

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res * 2]
            inject_noise = "small" not in self.opt.netG
            convs.append(
                StyledConv(in_channel, out_channel, 3, upsample=True, blur_kernel=blur_kernel, inject_noise=inject_noise)
            )
            cur_res = cur_res * 2

        convs.append(ConvLayer(channels[cur_res], 3, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        return self.convs(input)


class StyleGAN2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, opt=None):
        super().__init__()
        self.opt = opt
        self.encoder = StyleGAN2Encoder(input_nc, output_nc, ngf, use_dropout, n_blocks, padding_type, no_antialias, opt)
        self.decoder = StyleGAN2Decoder(input_nc, output_nc, ngf, use_dropout, n_blocks, padding_type, no_antialias, opt)

    def forward(self, input, layers=[], encode_only=False):
        feat, feats = self.encoder(input, layers, True)
        if encode_only:
            return feats
        else:
            fake = self.decoder(feat)

            if len(layers) > 0:
                return fake, feats
            else:
                return fake
