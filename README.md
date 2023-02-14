# Real-and-Synthetic-Korean-Licence-Plate-Generator

This repository contains Python implementation of real and synthetic Korean licence plate (LP) registration numbers. First, the synthetic Korean LP numbers are generated and they are used as input to the Generative Adversarial Network (GAN) model to make real-life LP numbers with certain amount of distortions.

### Synthetic Korean Licence Plates Generation
The synthetic LP numbers are generated based on [the latest available online information](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_South_Korea). According to the information, there are six widely-used car LP types in South Korea:

* [Private (European-sized)](https://upload.wikimedia.org/wikipedia/commons/3/3d/Plak-Shakhsi-KOR.png);
* [Private (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/1/18/Plak-Shakhsi-335x155-KOR.png);
* [Commercial (European-sized)](https://upload.wikimedia.org/wikipedia/commons/e/e2/Plak-Tejari-KOR.png);
* [Commercial (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/6/6f/Plak-Tejari-335x170-KOR.png);
* [Private Cars Old-style (1973~2003)](https://upload.wikimedia.org/wikipedia/commons/9/9c/ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car_-_Daegu%281996-2004%29.jpg);
* [Private Cars Old-style (2004~2006)](https://en.wikipedia.org/wiki/File:ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car(2004-2006).jpg);

##### 🚗 Sample LPs 🚗
[Private European-sized (3 digit)](https://user-images.githubusercontent.com/50166164/218385697-113a1610-d3e0-4ccb-8212-8bc68556e4d9.jpg) | [Commercial European-sized](https://user-images.githubusercontent.com/50166164/218385792-7de1be1a-51e9-48a4-991f-9948382e8fb3.jpg)
:-:|:-:
<img src=https://user-images.githubusercontent.com/50166164/218385697-113a1610-d3e0-4ccb-8212-8bc68556e4d9.jpg height=150px> | <img src=https://user-images.githubusercontent.com/50166164/218385792-7de1be1a-51e9-48a4-991f-9948382e8fb3.jpg height=150px>
 
[Private North American-sized (3 digit)](https://user-images.githubusercontent.com/50166164/218386944-87f51541-5016-44c7-9d2d-0b45e073e621.jpg) | [Commercial North American-sized](https://user-images.githubusercontent.com/50166164/218386808-c14fd229-fb3f-4464-8859-1c6c0fd6b94f.jpg)
:-:|:-:
<img src=https://user-images.githubusercontent.com/50166164/218386944-87f51541-5016-44c7-9d2d-0b45e073e621.jpg height=150px> | <img src=https://user-images.githubusercontent.com/50166164/218386808-c14fd229-fb3f-4464-8859-1c6c0fd6b94f.jpg height=150px>

[Private North American-sized (2 digit)](https://user-images.githubusercontent.com/50166164/218628189-0dab45b8-ed2c-4bef-84da-00c42dccc786.jpg) | [Private North American-sized (2 digit)](https://user-images.githubusercontent.com/50166164/218628118-21eab9ea-7619-41e2-889c-311caf1c5a53.jpg)
:-:|:-:
<img src=https://user-images.githubusercontent.com/50166164/218628189-0dab45b8-ed2c-4bef-84da-00c42dccc786.jpg height=150px> | <img src=https://user-images.githubusercontent.com/50166164/218628118-21eab9ea-7619-41e2-889c-311caf1c5a53.jpg height=150px>


[Private Cars Old-style](https://user-images.githubusercontent.com/50166164/218387305-df52063b-c9e3-48e7-8ec2-f62b41edfb8c.jpg) | [Private Cars Old-style](https://user-images.githubusercontent.com/50166164/218387367-728251b9-db74-455b-8952-5db5d98133d6.jpg)
:-:|:-:
<img src=https://user-images.githubusercontent.com/50166164/218387305-df52063b-c9e3-48e7-8ec2-f62b41edfb8c.jpg height=150px> | <img src=https://user-images.githubusercontent.com/50166164/218387367-728251b9-db74-455b-8952-5db5d98133d6.jpg height=150px>





### Real Korean Licence Plates Generation
