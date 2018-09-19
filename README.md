# Deep Convolutional Generative Adversarial Network for generating fake Atari game images (Pytorch)

This project includes code for a Deep Convolutional Generative Adversarial Network that generates fake Atari game images. The idea of this project is based on chapter 3 of Maxim Lapan`s book called "Deep Reinforcement Learning Hands-On".

### Examples


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cpow-89/Cross-Entropy-metaheuristic-method-for-Reinforcement-Learning-Pytorch-/master/images/MountainCarContinuous-v0.gif "Trained Gan Tensorboard Output"

#### MountainCarContinuous-v0

![Trained Gan Tensorboard Output][image1]

### Dependencies

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Install Dependencies
numpy==1.14.2
gym==0.10.4
opencv-python==3.4.0.12
scipy==1.0.1
torch==0.4
torchvision==0.2.0
tensorboardX==1
tensorflow==1.7.0
tensorboard==1.7.0
matplotlib==2.2.2

### Instructions

You can run the project via main.py file through the console.



open the console and run: python main.py -c "your_config_file".json 
optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "Atari_Gan.json" 


You can view the results via tensorboardX:
tensorboard --logdir=path/to/log-directory
