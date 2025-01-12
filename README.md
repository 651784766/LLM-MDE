"# LLM-MDE" 
# Deep-learning-based pyramid-transformer for localized porosity analysis of hot-press sintered ceramic paste

![pytorch](https://img.shields.io/badge/pytorch-v1.10-green.svg?style=plastic)
![wandb](https://img.shields.io/badge/wandb-v0.12.10-blue.svg?style=plastic)
![scipy](https://img.shields.io/badge/scipy-v1.7.3-orange.svg?style=plastic)

<!-- ![presentation](https://i.ibb.co/rbySmMc/DL-FOD-POSTER-1.png) -->

<p align="center">
  <img src="images/pull_figure.png"/>
</p>

<!-- > Input image taken from: https://koboguide.com/how-to-improve-portrait-photography/ -->

## Abstract

<!-- Recent works have shown that in the real world, humans
rely on the image obtained by their left and right eyes in order to estimate depths of surrounding objects. Thus, -->
>Scanning Electron Microscope (SEM) is a crucial tool for studying microstructures of ceramic materials. However, the current practice heavily relies on manual efforts to extract porosity from SEM images. To address this issue, we propose PSTNet (Pyramid Segmentation Transformer Net) for grain and pore segmentation in SEM images, which merges multi-scale feature maps through operations like recombination and upsampling to predict and generate segmentation maps.


## :pushpin: Requirements

Run: ``` pip install -r requirements.txt ```

## :rocket: Running the model

You can first download one of the models from the model:

### :bank: Model zoo

Get the links of the following models:

+ [```PST-large.p```]
+ Other models coming soon...

And put the ```.p``` file into the directory ```models/```. After that, you need to update the ```config.json```  according to the pre-trained model you have chosen to run the predictions (this means that if you load a depth-only model, then you have to set ```type``` to ```depth``` for example ....

### :dart: Run a prediction

Put your input images (that have to be ```.png``` or ```.jpg```) into the ```input/``` folder. Then, just run ```python run.py``` and you should get the depth maps as well as the segmentation masks in the ```output/``` folder.


## :hammer: Training

### :wrench: Build the dataset

Our model is trained on a combination of
+ [Al2O3 Dataset](https://github.com/Howtocreateaname/DL-based-porosity-characterization)
+ [Y2O3 Dataset]()

### :pencil: Configure ```config.json```

Specific configurations are given in the paper

### :nut_and_bolt: Run the training script
After that, you can simply run the training script: ```python train.py```


## :scroll: Citations

Our work is based on the research of Ranflt et al. Unlike them, we will focus on ceramic semantic segmentation. Thank you for your support!
