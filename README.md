# 3D Deep learning, Classification and Reconstruction/Shape Completion
This project was made in the course Advanced Machine Learning with Neural Networks, from Chalmers. I was given the dataset and objective, and I chose to implement 2 novel deep learning models (PointNet and VoxNet) for 3D object classification and compare them. Furthermore, I was curious to see if I could extend the VoxNet architecture to handle shape completion/reconstruction, based on the contents of the course. Thus, I ended up creating a U-net style model able to complete partial objects with an IoU score of 68.55%.

# Repo structure
## Classification
- /PointNet
- /VoxNet

## Shape Completion
- /Unet-style_VoxNet
  - VoxNetAE.py has no skip connections
  - VoxNetAE_Skip.py has skip connections between each of the convolution blocks in the encoder to the decoder. 

## utils notebook for downloading dataset

## Short report
