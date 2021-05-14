# Semantic Segmentation on MIT ADE20K dataset in PyTorch

### Syncronized Batch Normalization on PyTorch
This module computes the mean and standard-deviation across all devices during training. We empirically find that a reasonable large batch size is important for segmentation. We thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions, please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details.

The implementation is easy to use as:
- It is pure-python, no C++ extra extension libs.
- It is completely compatible with PyTorch's implementation. Specifically, it uses unbiased variance to update the moving average, and use sqrt(max(var, eps)) instead of sqrt(var + eps).
- It is efficient, only 20% to 30% slower than UnsyncBN.

### Dynamic scales of input for training with multiple GPUs 
For the task of semantic segmentation, it is good to keep aspect ratio of images during training. So we re-implement the `DataParallel` module, and make it support distributing data to multiple GPUs in python dict, so that each gpu can process images of different sizes. At the same time, the dataloader also operates differently. 

<sup>*Now the batch size of a dataloader always equals to the number of GPUs*, each element will be sent to a GPU. It is also compatible with multi-processing. Note that the file index for the multi-processing dataloader is stored on the master process, which is in contradict to our goal that each worker maintains its own file list. So we use a trick that although the master process still gives dataloader an index for `__getitem__` function, we just ignore such request and send a random batch dict. Also, *the multiple workers forked by the dataloader all have the same seed*, you will find that multiple workers will yield exactly the same data, if we use the above-mentioned trick directly. Therefore, we add one line of code which sets the defaut seed for `numpy.random` before activating multiple worker in dataloader.</sup>

## Supported models
We split our models into encoder and decoder, where encoders are usually modified directly from classification networks, and decoders consist of final convolutions and upsampling. We have provided some pre-configured models in the ```config``` folder.

Encoder:
- MobileNetV2dilated
- ResNet18/ResNet18dilated
- ResNet50/ResNet50dilated
- ResNet101/ResNet101dilated
- HRNetV2 (W48)

Decoder:
- C1 (one convolution module)
- C1_deepsup (C1 + deep supervision trick)
- PPM (Pyramid Pooling Module, see [PSPNet](https://hszhao.github.io/projects/pspnet) paper for details.)
- PPM_deepsup (PPM + deep supervision trick)
- UPerNet (Pyramid Pooling + FPN head, see [UperNet](https://arxiv.org/abs/1807.10221) for details.)

## Environment
The code is developed under the following configurations.
- Hardware: >=4 GPUs for training, >=1 GPU for testing (set ```[--gpus GPUS]``` accordingly)
- Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0***
- Dependencies: numpy, scipy, opencv, yacs, tqdm

## Quick start: Test on an image using our trained model 
1. Here is a simple demo to do inference on a single image:
```bash
chmod +x demo_test.sh
./demo_test.sh
```
This script downloads a trained model (ResNet50dilated + PPM_deepsup) and a test image, runs the test script, and saves predicted segmentation (.png) to the working directory.

2. To test on an image or a folder of images (```$PATH_IMG```), you can simply do the following:
```
python3 -u test.py --imgs $PATH_IMG --gpu $GPU --cfg $CFG
```

## Training
1. Download the ADE20K scene parsing dataset:
```bash
chmod +x download_ADE20K.sh
./download_ADE20K.sh
```
2. Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, checkpoints by default are saved in folder ```ckpt```.
```bash
python3 train.py --gpus $GPUS --cfg $CFG 
```
- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

For example, you can start with our provided configurations: 

* Train MobileNetV2dilated + C1_deepsup
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Train ResNet50dilated + PPM_deepsup
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Train UPerNet101
```bash
python3 train.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```

3. You can also override options in commandline, for example  ```python3 train.py TRAIN.num_epoch 10 ```.


## Evaluation
1. Evaluate a trained model on the validation set. Add ```VAL.visualize True``` in argument to output visualizations as shown in teaser.

For example:

* Evaluate MobileNetV2dilated + C1_deepsup
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-mobilenetv2dilated-c1_deepsup.yaml
```

* Evaluate ResNet50dilated + PPM_deepsup
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet50dilated-ppm_deepsup.yaml
```

* Evaluate UPerNet101
```bash
python3 eval_multipro.py --gpus GPUS --cfg config/ade20k-resnet101-upernet.yaml
```
    
