# Real-and-Synthetic-Korean-Licence-Plate-Generator

This repository contains PyTorch implementation of real and synthetic Korean licence plate (LP) registration numbers. First, the synthetic Korean LP numbers are generated and they are used as input to the Generative Adversarial Network (GAN) model to make real-life LP numbers with certain amount of distortions.

### Virtual Environment Creation

```python

conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt

```

### Synthetic Korean Licence Plates Generation
The synthetic LP numbers are generated based on [the latest available online information](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_South_Korea). According to the information, there are six widely-used car LP types in South Korea:

* [Private (European-sized)](https://upload.wikimedia.org/wikipedia/commons/3/3d/Plak-Shakhsi-KOR.png);
* [Private (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/1/18/Plak-Shakhsi-335x155-KOR.png);
* [Commercial (European-sized)](https://upload.wikimedia.org/wikipedia/commons/e/e2/Plak-Tejari-KOR.png);
* [Commercial (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/6/6f/Plak-Tejari-335x170-KOR.png);
* [Private Cars Old-style (1973~2003)](https://upload.wikimedia.org/wikipedia/commons/9/9c/ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car_-_Daegu%281996-2004%29.jpg);
* [Private Cars Old-style (2004~2006)](https://en.wikipedia.org/wiki/File:ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car(2004-2006).jpg);

##### 🚗 Sample LPs 🚗

|Private European-sized (3 digit) | Private North American-sized (3 digit) | Private European-sized (2 digit) | Private North American-sized (2 digit) |
|       :----:       |     :----:        |         :----:         |        :----:         | 
| <img src=https://user-images.githubusercontent.com/50166164/218385697-113a1610-d3e0-4ccb-8212-8bc68556e4d9.jpg width=350px height=80px> | <img src=https://user-images.githubusercontent.com/50166164/218386944-87f51541-5016-44c7-9d2d-0b45e073e621.jpg width=200px height=120px> | <img src=https://user-images.githubusercontent.com/50166164/218628189-0dab45b8-ed2c-4bef-84da-00c42dccc786.jpg width=320px height=80px> | <img src=https://user-images.githubusercontent.com/50166164/218628118-21eab9ea-7619-41e2-889c-311caf1c5a53.jpg width=200px height=120px> |

| Commercial European-sized | Commercial North American-sized | Private Cars Old-style | Private Cars Old-style |
|       :----:       |     :-:        |         :---:         |        :---:         | 
| <img src=https://user-images.githubusercontent.com/50166164/218385792-7de1be1a-51e9-48a4-991f-9948382e8fb3.jpg width=260px height=80px> | <img src=https://user-images.githubusercontent.com/50166164/218386808-c14fd229-fb3f-4464-8859-1c6c0fd6b94f.jpg width=200px height=120px> | <img src=https://user-images.githubusercontent.com/50166164/218387305-df52063b-c9e3-48e7-8ec2-f62b41edfb8c.jpg width=200px height=120px> | <img src=https://user-images.githubusercontent.com/50166164/218387367-728251b9-db74-455b-8952-5db5d98133d6.jpg width=200px height=120px> |

##### :computer: Generate synthetic LPs from pre-defined file :computer:
```
python generate.py --data_path "path/to/csv_file" --save_path "path/to/save/synthetic_lps" --random=False --transformations=False --save=True
```
This script gets information about the LPs from pre-defined csv file, generates synthethic LPs, and saves them into the save_path.

##### :computer: Generate synthetic LPs (random generation) :computer:
```
python generate.py --save_path "path/to/save/synthetic_lps" --random=True --transformations=False --random=True --save=True --number_of_plates 100
```
This script randomly creates LP information, generates synthethic LPs from the randomly created information, and saves them into the save_path.

##### :computer: Create train dataset :computer:
```
python make_dataset.py --in_im_paths "path/to/generated/synthetic_lps" --out_im_paths "path/to/real-life/images" --trainA "path/to/copy/synthetic/images" --trainB "path/to/copy/real-life/images" --type "train or test depending on dataset type"
```

### Real Korean Licence Plates Generation

##### Train

After getting synthetic images, we train them using modified (more efficient and fast) [CUT GAN model](https://github.com/taesungp/contrastive-unpaired-translation) as follows:

```
python train.py --dataroot path/to/the/dataset --name name/of/the/trained/model --CUT_mode CUT/FastCUT
```
This script trains the model based on the "--CUT_mode" argument (CUT or FastCUT) using the given dataroot (the root should contain two folders, trainA and trainB, respectively) and saves the model outputs under "--name" (this is later used for testing purposes) model name.

##### Inference

```
python test.py --dataroot path/to/the/dataset --name name/of/the/trained/model --CUT_mode CUT/FastCUT --phase test
```
This script conducts inference with the pretrained model (choose the model using "--name" argument) based on the "--CUT_mode" argument (CUT or FastCUT) using the given test dataroot (the root should contain two folders, testA and testB, respectively). The inference results can be found at ./results/name/train_latest/...

Generated sample LPs can be seen below:

![first](https://user-images.githubusercontent.com/50166164/219285736-0a9e8771-d05b-4da2-973b-7eef434610e8.png)
![second](https://user-images.githubusercontent.com/50166164/219285778-9b32996a-ae7a-4456-adda-359ffff46ebf.png)
![third](https://user-images.githubusercontent.com/50166164/219287897-0734fce3-1df8-4899-868e-c94e2cbb6898.png)
![fourth](https://user-images.githubusercontent.com/50166164/219287922-e8a6ec2f-8041-4972-a795-e023338bd894.png)
![fifth](https://user-images.githubusercontent.com/50166164/219287934-9f485ecc-616e-4805-9808-4dfb667979d6.png)
![sixth](https://user-images.githubusercontent.com/50166164/219287955-ec52f52d-ae88-4f29-b3c7-5466f454b76c.png)









