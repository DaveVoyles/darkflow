## Intro
[![Build Status](https://travis-ci.org/thtrieu/darkflow.svg?branch=master)](https://travis-ci.org/thtrieu/darkflow) [![codecov](https://codecov.io/gh/thtrieu/darkflow/branch/master/graph/badge.svg)](https://codecov.io/gh/thtrieu/darkflow)

##### Real-time object detection and classification. 

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. Darkflow is a TensorFlow implementation of Darknet, which allows you to write your code in Python.

-------------------------------------

Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).
Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). 

See demo below or see on [this imgur](http://i.imgur.com/EyZZKAA.gif)

<p align="center"> <img src="demo.gif"/> </p>

### Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

### Install Darkflow

There are 3 methods of doing so, and you only need to do **one**. I've found that **pip install -e .** worked best. 

* Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

* Install with pip globally
    ```
    pip install .
    ```
    
* Build the Cython extensions in place. **NOTE:** *If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.*
    ```
    python3 setup.py build_ext --inplace
    ```
    
### Downlod the weights

A weight is the strength of the connection between nodes in a neural network. If you increase the input then how much influence does it have on the output. Weights near zero mean changing this input will not change the output. Weights and biases are the learnable parameters of your model. The values of these parameters before learning starts are initialised randomly (this stops them all converging to a single value). Then, when presented with data during training, they are adjusted towards values that have correct output. This is what is currently in these different weight files.

These can grow to 100mb+ per file, so for that reason they are not included in the repository.  In case the weight file cannot be found on the [Darknet site](https://pjreddie.com/darknet/yolo/), the [author of Darkflow uploaded some of his here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include yolo-full and yolo-tiny of v1.0, tiny-yolo-v1.1 of v1.1 and yolo, tiny-yolo-voc of v2.

You will need to place all of the weights in the `bin/` folder. In the end, your structure should look like this:

```
|- darkflow-master/

|--- bin/

|------ yolo.weights

|------ tiny-yolo.weights

|------ yolo-tiny.weights

|------ tiny-yolo-v1.1.weights

|------ tiny-yolo-voc.weights

|------ yolo3.weights
```

## Parsing the annotations

Skip this if you are not training or fine-tuning anything (you simply want to forward flow a trained net)

For example, if you want to work with only 3 classes `tvmonitor`, `person`, `pottedplant`; edit `labels.txt` as follows

```
tvmonitor
person
pottedplant
```

And that's it. `darkflow` will take care of the rest. You can also set darkflow to load from a custom labels file with the `--labels` flag (i.e. `--labels myOtherLabelsFile.txt`). This can be helpful when working with multiple models with different sets of output labels. When this flag is not set, darkflow will load from `labels.txt` by default (unless you are using one of the recognized `.cfg` files designed for the COCO or VOC dataset - then the labels file will be ignored and the COCO or VOC labels will be loaded).

## Design the net

OPTIONAL: Skip this if you are working with one of the original configurations since they are already there. Otherwise, see the following example:

```python
...

[convolutional]
batch_normalize = 1
size = 3
stride = 1
pad = 1
activation = leaky

[maxpool]

[connected]
output = 4096
activation = linear

...
```

## Flowing the graph using `flow`

```bash
# Have a look at its options
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. Load .weights
flow --model cfg/v1/yolo-tiny.cfg --load bin/yolo-tiny.weights --savepb --verbalise
```

**NOTE:** If you see the error ```AssertionError: expect 64701556 bytes, found 180357512``` that means your .cfg and .weights files do not match up. Notice that we are using the `v1/tiny-yolo.cfg` file here, and NOT the `tiny-yolo.cfg` file in the `/cfg` folder. See [Mikeknapp's answer to this issue](https://github.com/thtrieu/darkflow/issues/620)

If all went well, you should see something similar to:

```
davevoyles@dv-dlvm-ubuntu:/tmp/mozilla_davevoyles0/darkflow-master$ flow --model cfg/v1/tiny-yolo.cfg --load bin/tiny-yolo.weights --savepb --verbalise
/anaconda/envs/py35/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters

/tmp/mozilla_davevoyles0/darkflow-master/darkflow/dark/darknet.py:54: UserWarning: ./cfg/tiny-yolo.cfg not found, use cfg/v1/tiny-yolo.cfg instead
  cfg_path, FLAGS.model))
Parsing cfg/v1/tiny-yolo.cfg
Loading bin/tiny-yolo.weights ...
Successfully identified 180357512 bytes
Finished in 0.00400090217590332s
Model has a VOC model name, loading VOC labels.

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 448, 448, 3)
 Load  |  Yep!  | scale to (-1, 1)                 | (?, 448, 448, 3)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 448, 448, 16)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 224, 224, 16)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 224, 224, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 32)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 112, 112, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 64)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 56, 56, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 128)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 28, 28, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 256)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 14, 14, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 7, 7, 512)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | flat                             | (?, 50176)
 Load  |  Yep!  | full 50176 x 256  linear         | (?, 256)
 Load  |  Yep!  | full 256 x 4096  leaky           | (?, 4096)
 Load  |  Yep!  | drop                             | (?, 4096)
 Load  |  Yep!  | full 4096 x 1470  linear         | (?, 1470)
-------+--------+----------------------------------+---------------
Running entirely on CPU
2018-06-06 22:21:03.991220: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-06-06 22:21:09.417456: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9340:00:00.0
totalMemory: 11.17GiB freeMemory: 11.10GiB
2018-06-06 22:21:09.417774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
Finished in 36.41262149810791s

Rebuild a constant version ...
2018-06-06 22:21:37.019380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-06-06 22:21:37.224531: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-06-06 22:21:37.224847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10765 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9340:00:00.0, compute capability: 3.7)
Done
```

Let's try running a new model, utilizing one of the .cfg files that came with Darkflow. You can replace `tiny-yolo` with any of the other config files found in the `cfg` or its folders.

```bash
# 2. To initialize a model, leave the --load option
# NOTE: The name is tiny-yolo.cfg now, and NOT tiny.yolo
flow --model cfg/tiny-yolo.cfg
```

This should return:

```
Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 416, 416, 3)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 416, 416, 16)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 208, 208, 16)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 208, 208, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 104, 104, 32)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 104, 104, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 52, 52, 64)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 52, 52, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 26, 26, 128)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 26, 26, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 13, 13, 256)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 512)
 Load  |  Yep!  | maxp 2x2p0_1                     | (?, 13, 13, 512)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Init  |  Yep!  | conv 3x3p1_1  +bnorm  leaky      | (?, 13, 13, 1024)
 Init  |  Yep!  | conv 1x1p0_1    linear           | (?, 13, 13, 425)
-------+--------+----------------------------------+---------------
```

```bash
# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
# this will print out which layers are reused, which are initialized
flow --model cfg/v1/yolo-tiny.cfg --load bin/yolo-tiny.weights

```

Which should return:

```
Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 448, 448, 3)
 Load  |  Yep!  | scale to (-1, 1)                 | (?, 448, 448, 3)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 448, 448, 16)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 224, 224, 16)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 224, 224, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 32)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 112, 112, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 64)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 56, 56, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 128)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 28, 28, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 256)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 14, 14, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 7, 7, 512)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | flat                             | (?, 50176)
 Load  |  Yep!  | full 50176 x 256  linear         | (?, 256)
 Load  |  Yep!  | full 256 x 4096  leaky           | (?, 4096)
 Load  |  Yep!  | drop                             | (?, 4096)
 Load  |  Yep!  | full 4096 x 1470  linear         | (?, 1470)
-------+--------+----------------------------------+---------------

```

All input images from default folder `sample_img/` are flowed through the net and predictions are put in `sample_img/out/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, images folder, etc.

### Utilizing the GPU
We can take advantage of the GPU by adding the ``gpu 1.0`` flag as well.

```bash
# Forward all images in sample_img/ using tiny yolo and 100% GPU usage
flow --model cfg/v1/yolo-tiny.cfg --load bin/yolo-tiny.weights --imgdir sample_img/ --gpu 1.0
```

The output will look similar to this:

```
Parsing cfg/v1/yolo-tiny.cfg
Loading bin/yolo-tiny.weights ...
Successfully identified 180357512 bytes
Finished in 0.004228353500366211s
Model has a VOC model name, loading VOC labels.

Building net ...
Source | Train? | Layer description                | Output size
-------+--------+----------------------------------+---------------
       |        | input                            | (?, 448, 448, 3)
 Load  |  Yep!  | scale to (-1, 1)                 | (?, 448, 448, 3)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 448, 448, 16)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 224, 224, 16)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 224, 224, 32)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 112, 112, 32)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 112, 112, 64)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 56, 56, 64)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 56, 56, 128)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 28, 28, 128)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 28, 28, 256)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 14, 14, 256)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 14, 14, 512)
 Load  |  Yep!  | maxp 2x2p0_2                     | (?, 7, 7, 512)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | conv 3x3p1_1    leaky            | (?, 7, 7, 1024)
 Load  |  Yep!  | flat                             | (?, 50176)
 Load  |  Yep!  | full 50176 x 256  linear         | (?, 256)
 Load  |  Yep!  | full 256 x 4096  leaky           | (?, 4096)
 Load  |  Yep!  | drop                             | (?, 4096)
 Load  |  Yep!  | full 4096 x 1470  linear         | (?, 1470)
-------+--------+----------------------------------+---------------
GPU mode with 1.0 usage
2018-06-08 15:45:17.711355: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
^[[2018-06-08 15:45:24.171648: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: bd3f:00:00.0
totalMemory: 11.17GiB freeMemory: 11.10GiB
2018-06-08 15:45:24.171694: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-06-08 15:45:24.454680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11441 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: bd3f:00:00.0, compute capability: 3.7)
2018-06-08 15:45:24.469000: E tensorflow/stream_executor/cuda/cuda_driver.cc:936] failed to allocate 11.17G (11996954624 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY
Finished in 10.985207319259644s

Forwarding 8 inputs ...
Total time = 19.503355026245117s / 8 inps = 0.4101858367052553 ips
Post processing 8 inputs ...
Total time = 0.2859928607940674s / 8 inps = 27.972726234451343 ips
```


json output can be generated with descriptions of the pixel location of each bounding box and the pixel location. Each prediction is stored in the `sample_img/out` folder by default. An example json array is shown below.
```bash
# Forward all images in sample_img/ using tiny yolo and JSON output.
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
```
JSON output:
```json
[{"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}},
{"label": "dog", "confidence": 0.32, "topleft": {"x": 71, "y": 263}, "bottomright": {"x": 193, "y": 353}},
{"label": "horse", "confidence": 0.76, "topleft": {"x": 412, "y": 109}, "bottomright": {"x": 592,"y": 337}}]
```
 - label: self explanatory
 - confidence: somewhere between 0 and 1 (how confident yolo is about that detection)
 - topleft: pixel coordinate of top left corner of box.
 - bottomright: pixel coordinate of bottom right corner of box.

## Training new model

Training is simple as you only have to add option `--train`. Training set and annotation will be parsed if this is the first time a new configuration is trained. To point to training set and annotations, use option `--dataset` and `--annotation`. A few examples:

```bash
# Initialize yolo-new from tiny-yolo, then train the net on 100% GPU:
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0

# Completely initialize yolo-new and train it with ADAM optimizer
flow --model cfg/yolo-new.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `ckpt/`. To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darkflow` will load the most recent save by parsing `ckpt/checkpoint`.

```bash
# Resume the most recent checkpoint for training
flow --train --model cfg/yolo-new.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolo-new.cfg --load 1500

# Fine tuning tiny-yolo from the original one
flow --train --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

Example of training on Pascal VOC 2007:
```bash
# Download the Pascal VOC dataset:
curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# An example of the Pascal VOC annotation format:
vim VOCdevkit/VOC2007/Annotations/000001.xml

# Train the net on the Pascal dataset:
flow --model cfg/yolo-new.cfg --train --dataset "~/VOCdevkit/VOC2007/JPEGImages" --annotation "~/VOCdevkit/VOC2007/Annotations"
```

### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference `tiny-yolo-voc-3c.cfg` (It is crucial that you leave the original `tiny-yolo-voc.cfg` file unchanged, see below for explanation).

2. In `tiny-yolo-voc-3c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `tiny-yolo-voc-3c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In our case, `labels.txt` will contain 3 labels.

    ```
    label1
    label2
    label3
    ```
5. Reference the `tiny-yolo-voc-3c.cfg` model when you train.

    `flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images`


* Why should I leave the original `tiny-yolo-voc.cfg` file unchanged?
    
    When darkflow sees you are loading `tiny-yolo-voc.weights` it will look for `tiny-yolo-voc.cfg` in your cfg/ folder and compare that configuration file to the new one you have set with `--model cfg/tiny-yolo-voc-3c.cfg`. In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.


## Camera/video file demo

For a demo that entirely runs on the CPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```

For a demo that runs 100% on the GPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```

To use your webcam/camera, simply replace `videofile.avi` with keyword `camera`.

To save a video with predicted bounding box, add `--saveVideo` option.

## Using darkflow from another python application

Please note that `return_predict(img)` must take an `numpy.ndarray`. Your image must be loaded beforehand and passed to `return_predict(img)`. Passing the file path won't work.

Result from `return_predict(img)` will be a list of dictionaries representing each detected object's values in the same format as the JSON output listed above.

```python
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```


## Save the built graph to a protobuf file (`.pb`)

```bash
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolo-new.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
```
When saving the `.pb` file, a `.meta` file will also be generated alongside it. This `.meta` file is a JSON dump of everything in the `meta` dictionary that contains information nessecary for post-processing such as `anchors` and `labels`. This way, everything you need to make predictions from the graph and do post processing is contained in those two files - no need to have the `.cfg` or any labels file tagging along.

The created `.pb` file can be used to migrate the graph to mobile devices (JAVA / C++ / Objective-C++). The name of input tensor and output tensor are respectively `'input'` and `'output'`. For further usage of this protobuf file, please refer to the official documentation of `Tensorflow` on C++ API [_here_](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html). To run it on, say, iOS application, simply add the file to Bundle Resources and update the path to this file inside source code.

Also, darkflow supports loading from a `.pb` and `.meta` file for generating predictions (instead of loading from a `.cfg` and checkpoint or `.weights`).
```bash
## Forward images in sample_img for predictions based on protobuf file
flow --pbLoad built_graph/yolo.pb --metaLoad built_graph/yolo.meta --imgdir sample_img/
```
If you'd like to load a `.pb` and `.meta` file when using `return_predict()` you can set the `"pbLoad"` and `"metaLoad"` options in place of the `"model"` and `"load"` options you would normally set.

That's all.
