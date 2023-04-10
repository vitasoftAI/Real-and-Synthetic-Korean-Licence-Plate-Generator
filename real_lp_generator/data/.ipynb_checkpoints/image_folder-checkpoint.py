"""

A modified image folder class of the official PyTorch image folder.

"""

# Import libraries
import torch.utils.data as data
from PIL import Image
import os
import os.path

# Initialize a list with proper image extensions
im_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

def is_image_file(fn):
    
    """
    
    This function checks filetype of the input file and returns True if the filetype is in the proper image extensions list.
    
    Argument:
        filename - name of the image file.
    
    """

    return any(fn.endswith(im_ext) for im_ext in im_exts)


def make_dataset(dir, max_dataset_size=float("inf")):
    
    """
    
    This function creates dataset by getting root folder and maximum number for dataset size.
    
    Arguments:
        
        dir - root folder with images, str.
        max_dataset_size - maximum number for the dataset, int.
    
    """
    
    ims = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir
    
    # Go through the directory
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        
        # Go through the file names in the root folder
        for fname in fnames:
            # Get image path
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # Add image path to the list
                ims.append(path)
    
    # Return a list with images based on max_dataset_size
    return ims[:min(max_dataset_size, len(ims))]

def default_loader(path):
    
    """
    
    This function loads an image from the given path.
    
    Argument:
        path - path to an image.
        
    Output:
        PIL image.
    
    """
    
    # Read image and convert it to RGB
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    
    """
    
    This class reads an image from the given path and returns transformed image.
    
    Arguments:
        
        root - path to the folder with images, str;
        transform - transformations to be applied;
        return_paths - option to return the path of an image, bool;
        loader - loader to load images from the root.
        
    Output:
        
        path - image path;
        img - transformed image.
    
    """

    def __init__(self, root, transform = None, return_paths = False, loader = default_loader):
        
        # Get image paths
        imgs = make_dataset(root)
        
        # If no image is in the root
        if len(imgs) == 0: raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(im_exts)))

        # Set class variables
        self.root, self.imgs, self.transform, self.return_paths, self.loader = root, imgs, transform, return_paths, loader

    def __getitem__(self, index):
        
        """
    
        This function reads image and applies transformations (if needed) and returns an image and path (if needed) based on the given index.

        Argument:
            path - path to an image.

        Output:
            PIL image.

        """
        
        # Get an image path
        path = self.imgs[index]
        # Load an image from the image path
        img = self.loader(path)
        # Apply transformations
        if self.transform is not None: img = self.transform(img)
        
        # Return image and path
        if self.return_paths: return img, path
        else: return img

    def __len__(self):
        
        """
    
        This function returns length of the dataset.

        """
        
        return len(self.imgs)
