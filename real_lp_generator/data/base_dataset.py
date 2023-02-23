"""

This module implements an abstract base class (ABC) 'BaseDataset' for datasets. 
It also includes common transformation functions (e.g., get_transform, __scale_width), 
which can be later used in subclasses.

"""

# Import libraries
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torchvision.transforms.functional as F
import numpy as np

class SquarePad:
    
    """ 
    
    Gets a rectangle shaped image and and creates a square image with padding.
    
    Argument:
    
    image - a rectangle shaped image.
    
    """
    
    def __call__(self, image):
        
        # Get width and height of the image
        w, h = image.size
        
        # Get max values of the width and height
        max_wh = np.max([w, h])
        hp = int((max_wh - w)/2)
        hp_rem = (max_wh - w)%2
        vp = int((max_wh - h)/2)
        vp_rem = (max_wh - h)%2
        
        # Pad the image
        padding = (hp, vp, hp+hp_rem, vp+vp_rem)
        
        return F.pad(image, padding, 255, 'constant')

class BaseDataset(data.Dataset, ABC):
    
    """ 
    
    An abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    
    """

    def __init__(self, opt):
        
        """
        
        Initializes the class; saves the options in the class.

        Arguments:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            
        """
        
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        
        """
        
        Adds new dataset-specific options, and rewrites default values for existing options.

        Arguments:
        
            parser - original option parser;
            is_train - whether training phase or test phase.

        Returns:
            
            the modified parser.
            
        """
        return parser

    @abstractmethod
    def __len__(self):
        
        """
        
        Returns the total number of images in the dataset.
        
        """
        
        return 0

    @abstractmethod
    def __getitem__(self, index):
        
        """ 
        
        Returns a data point and its metadata information.

        Arguments:
        
            index - a random integer for data indexing.

        Returns:
        
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        
        """
        
        pass

def get_params(opt, size):
    
    """
    
    Function to get parameters for transformation.
    
    Arguments:
    
        opt - options, str;
        size - pre-defined height and width, tuple.
    
    Returns:
    
        crop coordinates and flip option.
    
    """
    
    # Get width and height based on pre-defined size
    w, h = size
    
    # Set new width and height
    new_w, new_h = w, h
    
    # If preprocess has resize and crop set width and height to pre-defined load size
    if opt.preprocess == 'resize_and_crop': new_h = new_w = opt.load_size
    
    # If preprocess has scaled width and crop option set new width and height
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    
    # Get random integers for crop
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    
    # Return cropping coordinates flip option
    return {'crop_pos': (x, y), 'flip': random.random() > 0.5}

def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    
    """
    
    Function to get transformations list;
    
    Arguments:
    
        opt - options, str;
        params - parameters;
        grayscale - grayscale option, bool;
        method - name of the method for resizing;
        convert - conversion option, bool.
        
    Returns:
    
        tranformations list.

    """
    
    # Initialize transformations list
    
    transform_list = []
    
    # Add grayscale transformations list
    if grayscale: transform_list.append(transforms.Grayscale(1))
    
    # Add fixed size resize transformations list
    if 'fixsize' in opt.preprocess: transform_list.append(transforms.Resize(params["size"], method))
    
    # Add resize transformations list
    if 'resize' in opt.preprocess:
        
        # Get list of sizes based on load size option
        osize = [opt.load_size, opt.load_size]
        
        # Manage the case of gta2cityscapes
        if "gta2cityscapes" in opt.dataroot: osize[0] = opt.load_size // 2
            
        # Add square padding to the transformations list
        transform_list.append(SquarePad())
        
        # Add resize to the transformations list
        transform_list.append(transforms.Resize(osize, method))
        
    # Add scale to the transformations list
    elif 'scale_width' in opt.preprocess: transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
    elif 'scale_shortside' in opt.preprocess: transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))
    
    # Add zoom to the transformations list
    if 'zoom' in opt.preprocess: transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method))) if params is None else transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))
    
    # Add crop to the transformations list
    if 'crop' in opt.preprocess: transform_list.append(transforms.RandomCrop(opt.crop_size)) if (params is None or 'crop_pos' not in params) else transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
    
    # Add path to the transformations list
    if 'patch' in opt.preprocess: transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))

    # Add trim to the transformations list
    if 'trim' in opt.preprocess: transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))

    # Make square image
    if opt.preprocess == 'none': transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    # Flip option
    if not opt.no_flip: transform_list.append(transforms.RandomHorizontalFlip()) if (params is None or 'flip' not in params) else transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # Convert option
    if convert:
        # Convert to tensor
        transform_list += [transforms.ToTensor()]
        
        # Grayscale image normalization
        if grayscale: transform_list += [transforms.Normalize((0.5,), (0.5,))]
        
        # RGB image normalization
        else: transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    
    """
    Function to make a square-shaped image.
    
    Arguments:
    
        img - input image, PIL image;
        base - base value, int;
        method - method to resize.
    
    Returns:
    
        resized image.    
    
    """
    
    # Get image size
    ow, oh = img.size
    
    # Get height to resize
    h = int(round(oh / base) * base)
    
    # Get width to resize
    w = int(round(ow / base) * base)
    
    # If new and old h and width are the same return the original input image
    if h == oh and w == ow: return img

    return img.resize((w, h), method)

def __random_zoom(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC, factor=None):
    
    """
    
    A function to zoom an image.
    
    Arguments:
    
        img - image to be zoomed, PIL image;
        target_width - desired width to be zoomed;
        crop_width - width of the crop, int;
        method - resize method;
        factor - factor to zoom, tuple.
        
    Return:
    
        img - zoomed image
    
    """
    
    # Get zoom level
    zoom_level = np.random.uniform(0.8, 1.0, size=[2]) if factor is None else (factor[0], factor[1])
    
    # Get image coordinates
    iw, ih = img.size
    
    # Get zoom coordinates
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    
    # Return the zoomed image
    return img.resize((int(round(zoomw)), int(round(zoomh))), method)

def __scale_shortside(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    
    """
    
    A function to scale the shortside of an image.
    
    Arguments:
    
        img - image to be scaled, PIL image;
        target_width - desired width to be scaled;
        crop_width - width of the crop, int;
        method - resize method;
        
    Return:
    
        img - scaled image.
    
    """
   
    # Get width and height of the image
    ow, oh = img.size
    
    # Get the shortside
    shortside = min(ow, oh)
    if shortside >= target_width: return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)

def __trim(img, trim_width):
    
    """
    
    A function to trim the image.
    
    Arguments:
    
        img - image to be trimmed, PIL image;
        trim_width - desired width to be trimmed;
        
    Return:
    
        img - trimmed image.
    
    """
    
    # Get width and height of the image
    ow, oh = img.size
    
    # Get start and end points
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    
    # Get start and end points
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    
    # Return cropped image
    return img.crop((xstart, ystart, xend, yend))

def __scale_width(img, target_width, crop_width, method=transforms.InterpolationMode.BICUBIC):
    
    """
    
    A function to scale the width of an image.
    
    Arguments:
    
        img - image to be scaled, PIL image;
        target_width - desired width to be scaled;
        crop_width - width of the crop, int;
        method - resize method;
        
    Return:
    
        img - scaled image.
    
    """
    
    # Get width and heigh of the image
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    
    """
    
    A function to crop an image.
    
    Arguments:
    
        img - image to be scaled, PIL image;
        pos - desired width to be scaled;
        crop_width - width of the crop, int;
        method - resize method;
        
    Return:
    
        img - cropped image.
    
    """
    
    # Get width and height of an image
    ow, oh = img.size
    
    # Set the coordinates
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    
    return img

def __patch(img, index, size):
    
    """
    
    A function to get patch of an image.
    
    Arguments:
    
        img - image to be scaled, PIL image;
        index - index;
        size - desired size, tuple;
        
    Return:
    
        img - patched image.
    
    """
    
    # Get width and height of an image
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    
    # Start coordinates
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    
    # Return cropped image
    return img.crop((gridx, gridy, gridx + size, gridy + size))

def __flip(img, flip):
    
    """
    
    A function to flip an image.
    
    Arguments:
    
        img - image to be flipped, PIL image;
        flip - flip option, bool;
        
    Return:
    
        img - flipped (or not flipped) image.
    
    """
    
    if flip: return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    return img

def __print_size_warning(ow, oh, w, h):
    
    
    """
    
    Print warning information about image size (print only once).
    
    Arguments:
        
        ow - loaded image width;
        oh - loaded image height;
        w - adjusted image width;
        h - adjusted image height.
    
    """
    
    if not hasattr(__print_size_warning, 'has_printed'):
        
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        
        __print_size_warning.has_printed = True
