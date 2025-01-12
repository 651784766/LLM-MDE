# Large Language Models Can Understand Depth from Monocular Images

![pytorch](https://img.shields.io/badge/pytorch-v1.10-green.svg?style=plastic)
![wandb](https://img.shields.io/badge/wandb-v0.12.10-blue.svg?style=plastic)
![scipy](https://img.shields.io/badge/scipy-v1.7.3-orange.svg?style=plastic)

<p align="center">
  <img src="images/pull_figure.png"/>
</p>

<!-- > Input image taken from: https://koboguide.com/how-to-improve-portrait-photography/ -->

## Abstract

<!-- Recent works have shown that in the real world, humans
rely on the image obtained by their left and right eyes in order to estimate depths of surrounding objects. Thus, -->
>Monocular depth estimation is a critical function in computer vision applications. This paper shows that large language models can effectively interpret depth with minimal supervision, using efficient resource utilization and a consistent neural network architecture. We introduce LLM-MDE, a multimodal framework that deciphers depth through language comprehension. Specifically, LLM-MDE employs two main strategies to enhance the pretrained LLM's capability for depth estimation: cross-modal reprogramming and an adaptive prompt estimation module. These strategies align vision representations with text prototypes and automatically generate prompts based on monocular images, respectively. Comprehensive experiments on real-world MDE datasets confirm the effectiveness and superiority of LLM-MDE, which excels in few/zero-shot tasks while minimizing resource use.


## :pushpin: Requirements

Run: ``` pip install -r requirements.txt ```

## :rocket: Running the model

You can first download one of the models from the model:

### :bank: Model zoo

Get the links of the following models:

+ Other models coming soon...

## :hammer: Training

### :wrench: Build the dataset

Our model is trained on a combination of
+ [Raw NYU Dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)


### :pencil: Configure ```arguments_train_nyu.txt```

Specific configurations are given in the paper

### :nut_and_bolt: Run the training script
After that, you can simply run the training script: ```python main.py .\configs\arguments_train_nyu.txt```


## :scroll: Citations

Our work is based on the research of Ranflt et al. Thank you for your support!
