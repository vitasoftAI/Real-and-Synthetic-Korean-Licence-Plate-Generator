# Import functions from utils
from utils import *

class PlateGenerator:
    
    """
    
    This class gets a path to save the generated images, random and transformations options and generates LPs.
    
    Arguments:
    
        save_path          - a path to save generated images, str;
        random             - randomness option, bool;
        transformations    - transformations option, bool.
    
    """
    
    def __init__(self, save_path, random, transformations):
        
        # Set class variables
        self.save_path, self.random, self.transformations = save_path, random, transformations
        
        # Initialize plate types list
        self.plate_types = ["basic_europe", "basic_north", "commercial_europe", "commercial_north", "green_old", "green_basic"]

        # Basic nums and chars0
        self.num_ims, self.num_lists = load("digits/digits_white/")
        self.char_ims, self.char_lists = load("letters/letters_white/")

        # Yellow nums and chars
        self.num_ims_yellow, self.num_lists_yellow = load("digits/digits_yellow/")
        self.char_ims_yellow, self.char_lists_yellow = load("letters/letters_yellow/")
        self.regions_yellow, self.regions_lists_yellow = load("regions/regions_yellow/")
       
        # Green nums and chars
        self.num_ims_green, self.num_lists_green = load("digits/digits_green/")
        self.char_ims_green, self.char_lists_green = load("letters/letters_green/")
        self.regions_green, self.regions_lists_green = load("regions/regions_green/")
        
       
    def preprocessing(self, plate, random, plate_types):
        
        """
        
        Preprocessing function: gets plate, randomness option, and plate types and returns three digit option, plate, plate_type, digits and name of the region.
        
        Arguments:
        
            plate          - LP, str;
            random         - randomness option, bool;
            plate_types    - types of plates, list.
        
        """
        
        # Random option True
        if random:
            plate_type = plate_types[int(np.random.choice(np.arange(0, len(plate_types)), p=[0.33, 0.32, 0.15, 0.15, 0.03, 0.02]))]
            init_digit_types = ["three", "two"]
            init_digit = init_digit_types[int(np.random.choice(np.arange(0, len(init_digit_types)), p=[0.4, 0.6]))]
            three_digit = True if init_digit == "three" else False

            plate = "경기01마0101" if plate_type in ["commercial_europe", "commercial_north", "green_old"] else "01마0000"
            if plate_type in ["commercial_europe", "commercial_north", "green_old", "green_basic"]: three_digit = False
        else:
            if plate[0].isalpha(): three_digit, plate_type = False, "commercial_europe"
            elif plate[0].isdigit():
                three_digit, plate_type = True if len(plate) > 7 else False, "basic_europe"
        
        if plate_type in ["commercial_north", "commercial_europe", "green_old"]:
            
            split = os.path.splitext(os.path.basename(plate))[0]
            region_name, digits = split[:2], split[2:]
            
            return three_digit, digits, plate_type, region_name
        
        else: return three_digit, plate, plate_type, None
    
    def assertion(self, region_name, region_names):
        
        """
        
        Gets a region name along with region names list and checks whether the region name in the list of regions.
        
        Arguments:
        
            region_name  - a name of the region in the plate, str;
            region_names - names of the regions, list;
        
        """
        
        assert region_name != None, "Please insert a region name"
        assert region_name in [os.path.basename(region) for region in region_names], f"Please choose one of these regions: {[os.path.basename(region) for region in region_names]}"
    
    def generation(self, plate, save, plate_type, num, region_name):
        
        """
        
        Gets plate, save option, plate type, number, and region name and generates synthethic plate(s).
        
        Arguments:
        
            plate        - LP with digits and letters, str;
            save         - save option, bool;
            num          - number of LPs to be generated, int;
            region_name  - name of a region, str.        
        
        """
        
        # Iterate based on the pre-defined number 
        for _ in range(num):
            plate_path, num_list, num_size, num_size_2, init_size, char_list, regions, num_ims, char_size, char_ims, region_size, all_regions, plate_size = "plates/plate_white.jpg", self.num_lists, (56, 83), None, (13, 36), self.char_lists, None, self.num_ims, (60, 83), self.char_ims, None, self.regions_lists_yellow, (520, 110)
            
            three_digit, plate, plate_type, region_name = self.preprocessing(plate, self.random, self.plate_types)
            self.assertion(region_name, self.regions_lists_yellow) if plate_type in ["commercial_north", "commercial_europe", "green_old"] else 0
            
            if plate_type == "basic_north": num_size, init_size, char_size, plate_size = (40, 83), (46, 10), (49, 70), (355, 155)
                
            elif plate_type == "commercial_north": plate_path, num_list, num_size, num_size_2, init_size, char_list, regions, num_ims, char_size, char_ims, region_size, all_regions, plate_size = "plates/plate_yellow.jpg", self.num_lists_yellow, (44, 60), (64, 90), (8, 76), self.char_lists_yellow, self.regions_yellow, self.num_ims_yellow, (64, 62), self.char_ims_yellow, (88, 60), self.regions_lists_yellow, (336, 170)
                
            elif plate_type == "commercial_europe": plate_path, num_list, char_list, regions, num_ims, char_ims, region_size = "plates/plate_yellow.jpg", self.num_lists_yellow,  self.char_lists_yellow, self.regions_yellow, self.num_ims_yellow, self.char_ims_yellow, (88, 60)
                
                
            elif plate_type == "green_old": plate_path, num_list, num_size, num_size_2, init_size, char_list, regions, num_ims, char_size, char_ims, region_size, all_regions, plate_size = "plates/plate_green.jpg", self.num_lists_green, (44, 60), (64, 90), (8, 76), self.char_lists_green, self.regions_green, self.num_ims_green, (64, 62), self.char_ims_green, (88, 60), self.regions_lists_yellow, (336, 170)
             
            elif plate_type == "green_basic": plate_path, num_list, num_size, num_size_2, init_size, char_list, regions, num_ims, char_size, char_ims, all_regions, plate_size = "plates/plate_green.jpg", self.num_lists_green, (60, 65), (80, 90), (8, 78), self.char_lists_green, self.regions_green, self.num_ims_green, (60, 65), self.char_ims_green, self.regions_lists_yellow, (336, 170)
            
            generate_plate(plate_path=plate_path, random=self.random,
                           plate=plate, num_size=num_size, num_size_2=num_size_2, 
                           num_list=num_list, init_size=init_size, 
                           char_list=char_list, regions=regions, three_digit = three_digit,
                           num_ims=num_ims, char_size=char_size, region_name=region_name,
                           char_ims=char_ims, label_prefix=plate_type,
                           save_path=self.save_path, region_size=region_size, all_regions=all_regions,
                           save_=save, plate_size=plate_size, transformations=self.transformations)
