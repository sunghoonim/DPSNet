# DPSNet

This codebase implements the system described in the paper:

DPSNet: End-to-end Deep Plane Sweep Stereo

[Sunghoon Im](https://sunghoonim.github.io/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/), [Steve Lin](https://www.microsoft.com/en-us/research/people/stevelin/), [In So Kweon](http://rcv.kaist.ac.kr/)

In ICLR 2019.

See the [paper](https://openreview.net/pdf?id=ryeYHi0ctQ) for more details. 

Please contact Sunghoon Im (sunghoonim27@gmail.com) if you have any questions.


## Requirements

Building and using requires the following libraries and programs

    Pytorch 0.4.0 (The codes for (0.3.0 or 1.0) are in the other brach)
    CUDA 9.0
    python 3.6.4
    scipy
    argparse
    tensorboardX
    progressbar2
    blessings
    path.py
    
The versions match the configuration we have tested on an ubuntu 16.04 system.

## Data Praparation 

Training data preparation requires the following libraries and programs

    opencv
    imageio
    joblib
    h5py
    lz4
    
1. Download DeMoN data (https://github.com/lmb-freiburg/demon)
2. Convert data

[Training data]
    
```
bash download_traindata.sh
python ./dataset/preparation/preparedata_train.py
```

[Test data]
    
```
bash download_testdata.sh
python ./dataset/preparation/preparedata_test.py
```
    
## Train
```
python train.py ./dataset/train/ --mindepth 0.5 --nlabel 64 --log-output
```

## Test
```
python test.py ./dataset/test/ --sequence-length 2 --output-print --pretrained-dps ./pretrained/dpsnet.pth.tar
```

## Test (ETH3D)
Download full results on ETH3D datasets from https://phuang17.github.io/DeepMVS/index.html and merge it with './dataset/ETH3D_results/' folder, which includes gt_cam
```
python test_ETH3D.py ./dataset/ETH3D_results/ --sequence-length 3 --output-print --pretrained-dps ./pretrained/dpsnet.pth.tar
```

## Updated result for Table 1
```
Paper (epoch 4) -> Update (epoch 10)

MVS    A.Rel  A.diff Sq.Rel  RMSE  R. log   a=1    a=2    a=3
Paper  0.0722 0.2095 0.0798 0.4928 0.1527 0.8930 0.9502 0.9760
Update 0.0813 0.2006 0.0971 0.4419 0.1595 0.8853 0.9454 0.9735

SUN3D  A.Rel  A.diff Sq.Rel  RMSE  R. log   a=1    a=2    a=3
Paper  0.1470 0.3234 0.1071 0.4269 0.1906 0.7892 0.9317 0.9672
Update 0.1469 0.3355 0.1165 0.4489 0.1956 0.7812 0.9260 0.9728

RGBD   A.Rel  A.diff Sq.Rel  RMSE  R. log   a=1    a=2    a=3
Paper  0.1538 0.5235 0.2149 0.7226 0.2263 0.7842 0.8959 0.9402
Update 0.1508 0.5312 0.2514 0.6952 0.2421 0.8041 0.8948 0.9268

Scenes A.Rel  A.diff Sq.Rel  RMSE  R. log   a=1    a=2    a=3
Paper  0.0558 0.2430 0.1435 0.7136 0.1396 0.9502 0.9726 0.9804
Update 0.0500 0.1515 0.1108 0.4661 0.1164 0.9614 0.9824 0.9880
```
