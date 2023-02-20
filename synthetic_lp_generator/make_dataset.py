# Import libraries
import os, cv2, argparse, shutil
import numpy as np
from glob import glob

# Add Arguments
parser = argparse.ArgumentParser('Make Dataset for CUT train')
parser.add_argument('--in_im_paths', help = 'Input Images Path', type = str, default='/home/ubuntu/workspace/bekhzod/imagen/Korean-license-plate-Generator/new_samples/aaa')
parser.add_argument('--out_im_paths', help='Output Images Path', type = str, default='/home/ubuntu/workspace/bekhzod/imagen/lp_recognition_cropped/val')
parser.add_argument('--trainA', help = 'trainA Path', type = str, default='/home/ubuntu/workspace/bekhzod/cut/datasets/kor_licence_plate_dataset_test/trainA')
parser.add_argument('--trainB', help='trainB Path', type = str, default='/home/ubuntu/workspace/bekhzod/cut/datasets/kor_licence_plate_dataset_test/trainB')
parser.add_argument('--type', help='Make train or test dataset (train and test)', type = str, default='train')
parser.add_argument('--num_imgs', dest='num_ims', help='number of images', type=int, default=1000000)

# Get Arguments
args = parser.parse_args()

def copy_files(im_paths, destination): 
    
    """
    
    Gets path to images and path to output images and copies images to the destination path.
    
    Arguments:
    im_paths - path to the images to be copied;
    destination - path to the directory to copy the images.
    
    """
    
    # Go through every path in the input paths
    for file in im_paths:
        
        # Copy images to the specified train folder
        shutil.copy(file, destination)
        
def get_train_ims(ims_paths, im_files): 
    
    """
    
    Gets images paths along with image file names and returns sorted paths of the images;
    
    Arguments:
    ims_paths - paths of the images;
    im_files - list with appropriate image files.
    
    """
    
    return sorted(glob(f"{ims_paths}/*/*{[im_file for im_file in im_files]}"))
def get_test_ims(ims_paths, im_files): return sorted(glob(f"{ims_paths}/*{[im_file for im_file in im_files]}"))

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

im_files = [".jpg", ".png", ".jpeg"]
input_im_paths = get_train_ims(args.in_im_paths, im_files)
print(f"There are {len(input_im_paths)} synthetic images!")
if args.type == "test": output_im_paths = get_train_ims(args.in_im_paths, im_files)
else: output_im_paths = get_test_ims(args.out_im_paths, im_files)
print(f"There are {len(output_im_paths)} original images!")
num_ims = min(args.num_ims, len(input_im_paths))

os.makedirs(args.trainA, exist_ok=True)
os.makedirs(args.trainB, exist_ok=True)

print("Copying images...")
copy_files(input_im_paths, args.trainA)
copy_files(output_im_paths, args.trainB)
print("Done!")
