from plate_generator import PlateGenerator
import argparse, os
import pandas as pd
import numpy as np

def run(args):
    
    generator = PlateGenerator(save_path=args.save_path, random=args.random, transformations=args.transformations)

    if args.random:
        generator.generation(save=args.save, num=args.number_of_plates, plate=None, plate_type=None, region_name=None)
        
    else:
        df = pd.read_csv(args.data_path)
        texts = [os.path.basename(filename) for filename in df["filename"]]
        for text in texts:
            generator.generation(text, args.save, num=1, plate_type=None, region_name=None)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Synthetic Korean Vehicle Registration Plates Generator")
    
    parser.add_argument("-dp", "--data_path", help = "Path to the csv file with plate numbers", type = str, default = "sample_lps.csv")
    parser.add_argument("-sp", "--save_path", help = "Directory to save generated images", type = str, default = "./synthetic_lp_samples/")
    parser.add_argument("-s", "--save", help = "Saving option", type = bool, default = True)
    parser.add_argument("-np", "--number_of_plates", help = "Number of images to generate", type = int, default = 50)
    parser.add_argument("-r", "--random", help = "Generate random plate numbers", type = bool, default = True)
    parser.add_argument("-t", "--transformations", help = "Apply transformations", type = bool, default = False)
    
    
    args = parser.parse_args()
    run(args) 
