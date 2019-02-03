# DPSNet

This codebase implements the system described in the paper:

DPSNet: End-to-end Deep Plane Sweep Stereo

[Sunghoon Im](https://sunghoonim.github.io/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/), [Steve Lin](https://www.microsoft.com/en-us/research/people/stevelin/), [In So Kweon](http://rcv.kaist.ac.kr/)

In ICLR 2019.

See the [paper](https://openreview.net/pdf?id=ryeYHi0ctQ) for more details. 

Please contact Sunghoon Im (sunghoonim27@gmail.com) if you have any questions.


## Requirements

Building and using requires the following libraries and programs

    Pytorch 0.3.1
    CUDA 9.0
    python 3.5.4
    scipy
    argparse
    tensorboardX
    progressbar2
    path.py
    
The versions match the configuration we have tested on an ubuntu 16.04 system.


## Train
python3 train.py $datapath --mindepth 0.5 --nlabel 64 --log-output

## Test
python3 test.py $datapath --output-print --sequence-length 2 --ttype test.txt --pretrained-dps pretrained/dpsnet.pth.tar
