# Real-and-Synthetic-Korean-Licence-Plate-Generator

This repository contains Python implementation of real and synthetic Korean licence plate (LP) registration numbers. First, the synthetic Korean LP numbers are generated and they are used as input to the Generative Adversarial Network (GAN) model to make real-life LP numbers with certain amount of distortions.

### Synthetic Korean Licence Plates Generation
The synthetic LP numbers are generated based on [the latest available online information](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_South_Korea). According to the information, there are six widely-used car LP types in South Korea:

* [Private (European-sized)](https://upload.wikimedia.org/wikipedia/commons/3/3d/Plak-Shakhsi-KOR.png);
* [Private (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/1/18/Plak-Shakhsi-335x155-KOR.png);
* [Commercial (European-sized)](https://upload.wikimedia.org/wikipedia/commons/e/e2/Plak-Tejari-KOR.png);
* [Commercial (North American-sized)](https://upload.wikimedia.org/wikipedia/commons/6/6f/Plak-Tejari-335x170-KOR.png);
* [Private Cars Old-style (with region)](https://upload.wikimedia.org/wikipedia/commons/9/9c/ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car_-_Daegu%281996-2004%29.jpg);
* [Private Passenger Cars Old-style (without region)](https://en.wikipedia.org/wiki/File:ROK_Vehicle_Registration_Plate_for_Private_Passenger_Car(2004-2006).jpg);


### Real Korean Licence Plates Generation
