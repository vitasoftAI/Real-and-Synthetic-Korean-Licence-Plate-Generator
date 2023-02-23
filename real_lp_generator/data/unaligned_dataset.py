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

    def __init__(self, opt, test=True):
        
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

        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range # qoldiq
        if self.test:
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
        else:
            plate_type = os.path.splitext(os.path.basename(A_path))[0].split("__")[1]
            out_path = glob(f"{os.path.join(self.dir_B, plate_type)}/*{[im_file for im_file in self.im_files]}")
            index_B = random.randint(0, len(out_path) - 1)
            B_path = out_path[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # print(A_path)
        # print(B_path)
        # if index % 100 == 0:
        # print(os.path.basename(A_path), os.path.basename(B_path))
        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
