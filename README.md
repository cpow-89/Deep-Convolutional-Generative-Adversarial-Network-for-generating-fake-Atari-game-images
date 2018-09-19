# Deep Convolutional Generative Adversarial Network for generating fake Atari game images (Pytorch)

This project includes code for a Deep Convolutional Generative Adversarial Network that generates fake Atari game images. The idea of this project is based on chapter 3 of Maxim Lapan`s book called "Deep Reinforcement Learning Hands-On".

### Examples


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cpow-89/Deep-Convolutional-Generative-Adversarial-Network-for-generating-fake-Atari-game-images/master/asserts/Bildschirmfoto%202018-09-12%2023_26_33.png "Trained Gan Tensorboard Output"

#### Example Output for Pong and Breakout

![Trained Gan Tensorboard Output][image1]

### Dependencies

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Install Dependencies
numpy==1.14.2 <br>
gym==0.10.4 <br>
opencv-python==3.4.0.12 <br>
scipy==1.0.1 <br>
torch==0.4 <br>
torchvision==0.2.0 <br>
tensorboardX==1 <br>
tensorflow==1.7.0 <br>
tensorboard==1.7.0 <br>
matplotlib==2.2.2 <br>

### Instructions

You can run the project via main.py file through the console.



open the console and run: python main.py -c "your_config_file".json 
optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "Atari_Gan.json" 


You can view the results via tensorboardX: <br>
tensorboard --logdir=path/to/log-directory
