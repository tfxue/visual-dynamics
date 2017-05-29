#  Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks

This repository contains pre-trained models and demo code for the project 'Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks', published at the Conferece of Neural Information Processing Systems (**NIPS**) 2016.

http://visualdynamics.csail.mit.edu/

## Prerequisites

#### Torch
We use Torch 7 (http://torch.ch) for our implementation.

#### Imagemagick (optional)
We use the convert toolbox in ['Imagemagick'](http://www.imagemagick.com/) to generate gif image for visualization. See demo.lua about how to disable it.

## Installation
Our current release has been tested on Ubuntu 14.04.

#### Clone the repository
```sh
git clone https://github.com/tfxue/visual-dynamics.git
```
#### Download pretrained models (1.4GB) 
```sh
download_models.sh
``` 

#### Run test code

Run demo.lua. There are few options in demo.lua:

**useCuda**: Set to false if not using Cuda

**gpuId**: GPU device ID

**demo**: Set it to 'all' to run all demos. Set it to 'demo?' to run a specific demo

**modeldir, datadir, outputdirRoot**: directories that stores model files, input files, and output files

**createGIF**: Generate gif visualization. This requires Imagemagick. Set it to false if Imagemagick is not installed.


## Sample input & output

**Demo 1: sample future frames from a single image**
<table>
<tr>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo1/input.png" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo1/sample_1.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo1/sample_2.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo1/sample_3.gif" height="160"></td>
</tr>
<tr>
<td> Input image </td>
<td> Sample 1 </td>
<td> Sample 2</td>
<td> Sample 3 </td>
</tr>
</table>

**Demo 2: transfer motion from a source pair to a target image**
<table>
<tr>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo2/source.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo2/target1_im1.png" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo2/target1.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo2/target2_im1.png" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo2/target2.gif" height="160"></td>
</tr>
<tr>
<td> Source motion </td>
<td> Target image 1 </td>
<td> Transfered motion </td>
<td> Target image 2 </td>
<td> Transfered motion </td>
</tr>
</table>

**Demo 3: visualize selected dimensions of the latent representation**
<table>
<tr>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo3/out_newz_dim0752.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo3/out_newz_dim1746.gif" height="160"></td>
<td><img src="http://visualdynamics.csail.mit.edu/repo/output/demo3/out_newz_dim2195.gif" height="160"></td>
</tr>
<tr>
<td> Dimension 0752 </td>
<td> Dimension 1746 </td>
<td> Dimension 2195 </td>
</tr>
</table>

## Datasets we used

- Exercise dataset: [zip, 1.1GB](http://visualdynamics.csail.mit.edu/exercise_dataset.zip)

## Reference

    @inproceedings{visualdynamics16,   
        author = {Xue, Tianfan and Wu, Jiajun and Bouman, Katherine L and Freeman, William T},   
        title = {Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks},   
        booktitle = {NIPS},   
        year = {2016}
    }
    
For any questions, please contact Tianfan Xue (tfxue@mit.edu) and Jiajun Wu (jiajunwu@mit.edu).
