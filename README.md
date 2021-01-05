# Super-Resolution-VDSR


This project is HW4 for NCTU CS Selected Topics in Visual Recognition using Deep Learning.

- Goal: train a model to reconstruct a high-resolution image from a low-resolution image.
- Implementation of CVPR2016 Paper: "Accurate Image Super--Resolution Using Very Deep Convolutional Networks"([link](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf))
- Reference github: https://github.com/twtygqyy/pytorch-vdsr
- Training dataset: 291 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- GeForce GTX 1080 Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)

## Installation
All requirements should be detailed in [requirements.txt](https://github.com/sweiichen/tiny-pascal-voc/blob/main/requirements.txt). 
You have to create your own python virtual environment.
- python version: 3.7.7 
- cuda version: 10.1.243


```bash 
pip install -r requirements.txt
```
- To prepare hdf5 format data sample, it's necessary to install MATLAB.

## Dataset Preparation
The training dataset contains 291 high resolution images. Download it [here](https://drive.google.com/file/d/1bIowNL2X6zqfnmfcxqX3oPh2GIR9lxnr/view?usp=sharing)
To prepare hdf5 format training sample, run generate_train.m saved in `data` folder.
Before run the code, modifiy the folder path to the path where you save your training images, and the result will save in `train.h5` file.
It takes one to two hour to process, including data augmentaion with downsampling and rotation. Moreover, it generates multi-scale training data, and the training data is generated with Matlab Bicubic Interplotation.
 `train.h5` take up about 14.95 GB space.


## Training

```bash
usage: main_vdsr.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED] [--gpus GPUS]
               
optional arguments:
  -h, --help            Show this help message and exit
  --batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.01
  --step                Learning rate decay, Default: n=10 epochs
  --cuda                Use cuda
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.4
  --threads             Number of threads for data loader to use Default=1
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --gpus GPUS           gpu ids (default: 0)
```

To train the model, simply run the code as follow. After training process, the model weight will automatically save in `checkpoint/` folder.
```bash
python main_vdsr.py 
```

You might chage the bath size or number of cpu workers, according to you GPU memory size. Follow the code usage, run the code as:
```
python main_vdsr.py --batchSize 200 --threads 8
```

The expected training times are:

 GPUs  | workers| batchSize |Epoch | Training Time
------------ | ------------- | ------------- |--------------|--------
 1x TitanX   | 1 | 1000 |1 | 17 mins

 
 
When starting running the code, you can see the ouput like this.
```bash
Namespace(batchSize=750, clip=0.4, cuda=True, gpus='0', lr=0.001, momentum=0.9, nEpochs=25, pretrained='', resume='', start_epoch=1, step=10,
threads=1, weight_decay=0.0001)
=> use gpu id: '0'
Random Seed:  5686
===> Loading datasets
===> Building model
/opt/conda/lib/python3.7/site-packages/torch/nn/_reduction.py:43: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
===> Setting GPU
===> Setting Optimizer
===> Training
Epoch = 1, lr = 0.001
main_vdsr.py:114: UserWarning: torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.
  nn.utils.clip_grad_norm(model.parameters(),opt.clip)
===> Epoch[1](100/1483): Loss: 1.7368372679
===> Epoch[1](200/1483): Loss: 1.6014300585
===> Epoch[1](300/1483): Loss: 1.7985384464
===> Epoch[1](400/1483): Loss: 1.6790879965
===> Epoch[1](500/1483): Loss: 1.7204670906
===> Epoch[1](600/1483): Loss: 1.8001666069
===> Epoch[1](700/1483): Loss: 1.7282457352
===> Epoch[1](800/1483): Loss: 1.6729698181
===> Epoch[1](900/1483): Loss: 1.7117488384
===> Epoch[1](1000/1483): Loss: 1.6356124878
===> Epoch[1](1100/1483): Loss: 1.7666661739
===> Epoch[1](1200/1483): Loss: 1.6534382105
===> Epoch[1](1300/1483): Loss: 1.6242351532
===> Epoch[1](1400/1483): Loss: 1.8508431911
epoch_duration: 1236.8132841587067
.
.
```

### Load trained parameters
Pretrained model weight is located in` model/model_epoch_8.pth`,trained on 291 images with data augmentation and the initial learning rate is `1e-3`.
\
Performance(PSNR) on lecture testing dataset:
scale | VDSR(ours) | Bicubic(Pillow Image)|
------------ | ------------- |-----
 3x   | 24.434 | 25.03|
## Inference

```bash
usage: inference.py [-h] [--cuda] [--gpus GPUS] [--gt] [--model MODEL]
                    [--input INPUT] [--output OUTPUT] [--factor FACTOR]
                    [--gt_folder GT_FOLDER]

PyTorch VDSR

optional arguments:
  -h, --help            show this help message and exit
  --cuda                Use cuda?
  --gpus GPUS           gpu ids (default: 0)
  --gt                  Use ground truth?
  --model MODEL         path to model weight
  --input INPUT         path to input images folder
  --output OUTPUT       path to output images folder
  --factor FACTOR       upscale factor
  --gt_folder GT_FOLDER
                        path to groud truth images folder
```
To do inference, run the following code and set the pathes to the folders of  input and output images.
```bash
python inference.py --cuda --input {input folder} --output {output folder} --model {path to model weight}
```
### Evaluation

To evalutation, it is necessary to set path of ground truth folder, use command line as:
```bash
python inference.py --cuda --gt --gt_folder {path to the folder of ground truth images}--input {input folder} --output {output folder} --model {path to model weight} 
```
It shows the PSNR of super-resolutions images with Bicubic  interpolation and the VDSR model compare with ground truth images in the terminal as follow:
```bash
.
.
Evaluation of 02.png:
psnr for bicubic is 27.904158192812552dB
psnr for vdsr is 27.368261221021616dB
Evaluation of 13.png:
psnr for bicubic is 24.499636162996794dB
psnr for vdsr is 23.680194096304234dB
Evaluation of 12.png:
psnr for bicubic is 25.214626132023206dB
psnr for vdsr is 24.433161290801017dB
Average PSNR of VDSR:24.228791362119868
Average PSNR of Bicubic:25.03128972741009