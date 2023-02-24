import os.path
from glob import glob
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

class UnalignedDataset(BaseDataset):
    
    """
    
    This dataset class loads unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB', respectively.
    
    """

    def __init__(self, opt, test = True):
        
        """
        
        This functiin initializes the unaligned dataset class.

        Arguments:
            opt - options need to be a subclass of BaseOptions;
            test - test option, bool.
            
        """
        
        BaseDataset.__init__(self, opt)
        self.test = test
        
        # Set a path to the directory with images from domain A
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A') 
        
        # Set a path to the directory with images from domain B
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        # Set a list with proper image filetypes
        self.im_files = [".jpg", ".png", ".jpeg"]
        
        # Set a path to the directory with images from domain A and B for test purposes
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        # Get image paths from the abovementioned folders (domain A and B)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))   

    def __getitem__(self, index):
        
        """
        
        This function returns an image and its metadata information.

        Arguments:
            index - a random integer for data indexing, int.

        Output:
        
            a dictionary that contains A, B, A_paths and B_paths, where:
            
            A - an image in the input domain A, tensor;
            B - corresponding image for the input domain A in the target domain B, tensor;
            A_paths - path of the image in the input domain A, str;
            B_paths - path of the image in the input domain B, str;
            
        """
        
        # Set A_path variable and make sure it is in the range of len of A_path
        A_path = self.A_paths[index % len(self.A_paths)] 
        
        # Set B_path variable and make sure it is random every __getitem__call to avoid fixed pair of images
        B_path = self.B_paths[random.randint(0, len(self.B_paths) - 1)]
        
        # Read images from both domains
        A_img, B_img = Image.open(A_path).convert('RGB'), Image.open(B_path).convert('RGB')
        
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        # Get transformations
        transform = get_transform(modified_opt)
        
        # Apply transformations
        A, B = transform(A_img), transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        
        """
        
        This function returns the total number of images in the dataset.

        Because there are two datasets from two different domains, the function returns the max number of images in the datasets.
        
        """
        
        return max(len(self.A_paths), len(self.B_paths))
