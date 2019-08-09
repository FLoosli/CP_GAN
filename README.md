# CP_GAN
Implementation of a Cut and Paste GAN in keras.

Idea after the following paper by Tal Remez, Jonathan Huang and Matthew Brown  
https://arxiv.org/pdf/1803.06414.pdf

Required Versions for the project to work.

Tensorflow, 1.13.1  
numpy, 1.16.1  
keras, 2.2.4  
skimage, 0.14.2  
matplotlib, 3.0.3  
scipy, 1.2.1  
pycocotools 2.0  

For computing with the GPU additional installation is required, please refer to the respecting cuda installation on their website
as it depends on the GPU in use.
In usual circumstances, the CPU mode should always work.

Please check the path variables lead to the right directory.

The path for the data instances use to train the network should be configured correctly in the program files on the top of the file
These instances files do not contain the images used to train the network and only lead to the online saved images.
Consequently, you need a working internet connection to work with the images.
Otherwise no further installation should be required to access the data.  
http://cocodataset.org/#download
