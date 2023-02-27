# Import libraries
from plate_generator import PlateGenerator
import argparse, os
import pandas as pd
import numpy as np

def run(args):
    
    generator = PlateGenerator(save_path=args.save_path, random=args.random, transformations=args.transformations)

    if args.random:
        generator.generation(save=args.save, num=args.number_of_plates, plate=None)
        
    else:

        # Read data
        df = pd.read_csv(args.data_path)
        
        # Get LPs from the dataframe
        lps = [os.path.basename(fname) for fname in df["filename"]]
        for lp in lps:
            generator.generation(lp, args.save, 1)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Synthetic Korean Vehicle Registration Plates Generator")
    
    parser.add_argument("-dp", "--data_path", help = "Path to the csv file with plate numbers", type = str, default = "sample_lps.csv")
    parser.add_argument("-sp", "--save_path", help = "Directory to save generated images", type = str, default = "generated_samples/")
    parser.add_argument("-s", "--save", help = "Saving option", type = bool, default = True)
    parser.add_argument("-np", "--number_of_plates", help = "Number of images to generate", type = int, default = 100)
    parser.add_argument("-r", "--random", help = "Generate random plate numbers", type = bool, default = False)
    parser.add_argument("-t", "--transformations", help = "Apply transformations", type = bool, default = False)
    
    
    args = parser.parse_args()
    run(args) 
