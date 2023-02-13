import os, cv2
import numpy as np
from transformations import transform_plate

def get_label_and_plate(plate, row, col, num_size, num_ims, random, three_digit, plate_chars, label, num_list, char_num, temp, need_temp):
        
    if temp == "0": three_digit = True
    plate_int = num_list[int(np.random.randint(low=1, high=len(num_list), size=1)) if three_digit else int(np.random.randint(low=0, high=len(num_list), size=1))] if random else num_list[int(plate_chars[char_num])]
    plate[row:row + num_size[1], col:col + num_size[0], :] = cv2.resize(num_ims[str(plate_int)], num_size)
    col += num_size[0]
    
    return (plate, label + str(plate_int), col, plate_int) if need_temp else (plate, label + str(plate_int), col)
    
def write_partial(plate, label, num_list, num_ims, plate_chars, num_size, row, col, three_digit, random):
    
    for i in range(-4, 0):
        plate, label, col = get_label_and_plate(plate, row, col, num_size, num_ims, random, three_digit, plate_chars, label, num_list, i, None, False)
    
    return plate, label
    
def write(plate, label, num_list, num_ims, init_size, three_digit, char_list, plate_chars, num_size, num_size_2, char_ims, char_size, label_prefix, row, col, random):
    
    if label_prefix == "basic_north" and three_digit: col -= 20
    elif label_prefix == "basic_europe" and three_digit: col -= 15
    if label_prefix == "basic_north": row, col = row - 5, col + 17
    
    for i in range(2): 
        if i == 0: temp = None
        (plate, label, col, temp) = get_label_and_plate(plate, row, col, num_size, num_ims, random, three_digit, plate_chars, label, num_list, i, temp, True)
    
    if label_prefix == "commercial_europe": pass
    else:    
        if three_digit: plate, label, col = get_label_and_plate(plate, row, col, num_size, num_ims, random, three_digit, plate_chars, label, num_list, 2, None, False)

    if label_prefix in ["green_old", "commercial_north"]: row, col = 85, 5 

    # character 3
    plate_int = char_list[int(np.random.randint(low=0, high=len(char_list), size=1))] if random else (plate_chars[-5])
    label += str(plate_int)
    
    try:
        if label_prefix in ["basic_north"]: row += 5
        plate[row:row + char_size[1], col:col + char_size[0], :] = cv2.resize(char_ims[plate_int], char_size)
    except:
        print("\n!!!!!!!!!!!! FILE MISSING ERROR !!!!!!!!!!!!")
        print(f"Character {plate_chars[-5]} is missing!\n")

    if label_prefix == "basic_north": row, col = row - 5, col + 59
    elif label_prefix == "basic_europe": col += 85
    elif label_prefix == "green_basic": row, col = 75, 8
    elif label_prefix in ["commercial_north", "green_old"]: row, col = row - 13, col + 65
    elif label_prefix in ["commercial_europe"]: col += 70
    
    plate, label = write_partial(plate, label, num_list, num_ims, plate_chars, num_size_2, row, col, three_digit, random) if num_size_2 != None else write_partial(plate, label, num_list, num_ims, plate_chars, num_size, row, col, three_digit, random)
        
    return plate, label

def save(plate, save_path, transformations, label):
    
    if transformations: plate = transform_plate(plate)
    
    folder = label.split('__')[0]
    save_dir = os.path.join(save_path, folder)
    os.makedirs(save_dir, exist_ok = True)
    cv2.imwrite(os.path.join(save_dir, f"{label.split('__')[1]}__{folder}") + ".jpg", plate)
    print(f"Plate {label.split('__')[1]}__{folder}.jpg is saved to {save_dir}/!")

def load(files_path):
    
    chars_paths = sorted(os.listdir(files_path))
    ims, chars = {}, [] 

    for char_path in chars_paths:
        fname = os.path.splitext(char_path)[0]
        im = cv2.imread(os.path.join(files_path, char_path))
        ims[fname] = im
        chars.append(char_path[0:-4])
        
    return ims, chars

def preprocess(plate, plate_path, plate_size, label_prefix, init_size):
    
    plate_chars = [char for char in plate]
    plate = cv2.resize(cv2.imread(plate_path), plate_size)
    label = f"{label_prefix}__" 
    row, col = init_size[0], init_size[1]
    
    return plate_chars, plate, label, row, col
    
def generate_plate(plate_path, plate, plate_size, num_size, num_size_2, random, all_regions,
                   char_size, init_size, num_list, three_digit, char_list, num_ims, char_ims, 
                   regions, region_name, region_size, save_path, label_prefix, save_, transformations):
    
    plate_chars, plate, label, row, col = preprocess(plate, plate_path, plate_size, label_prefix, init_size)
    
    if random: region_name = all_regions[int(np.random.randint(low=0, high=len(all_regions), size=1))]

    if label_prefix == "commercial_europe":

        row, col = 10, 25
        to_crop = regions[region_name].shape[1] // 2
        plate[row:row + row * 4, col:col + col * 2, :] = cv2.resize(regions[region_name][:, 0:to_crop], (col * 2, row * 4))
        row += 45
        plate[row:row + row - 15, col:col + col * 2, :] = cv2.resize(regions[region_name][:, to_crop:], (col * 2, row - 15))
        row, col = 13, 100
        
    elif label_prefix in ["commercial_north", "green_old"]:
        plate[row:row + region_size[1], col:col + region_size[0], :] = cv2.resize(regions[region_name], region_size)
        col += region_size[0] + 8
        
    plate, label = write(plate=plate, label=label, num_list=num_list, num_ims=num_ims, random=random,
                         init_size=init_size, three_digit=three_digit, plate_chars=plate_chars, char_list=char_list,
                         num_size_2=num_size_2, char_ims=char_ims, char_size=char_size, 
                         label_prefix=label_prefix, row=row, num_size=num_size, col=col)

    if save_: save(plate=plate, save_path=save_path, transformations=transformations, label=label)
