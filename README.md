# MSTN
Efficient and Accurate Multi-scale Topological Network for Single Image Dehazing

### Requirements

* python3
* PyTorch
* NVIDIA GPU+CUDA
* numpy
* matplotlib
* tensorboardX(optional)
* scikit-image

### Datasets
Haze datasets
* [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/) [[RESIDE: A Benchmark for Single Image Dehazing](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1712.04143.pdf&sa=D&sntz=1&usg=AFQjCNHzdt3kMDsvuJ7Ef6R4ev59OFeRYA)]
* [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) [[NH-HAZE: An Image Dehazing Benchmark with
Non-Homogeneous Hazy and Haze-Free Images](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH_HAZE_IEEE.pdf)]
* [MiddleBury](https://vision.middlebury.edu/stereo/data/2014/) [[High-Resolution Stereo Datasets with
Subpixel-Accurate Ground Truth](https://elib.dlr.de/90624/1/ScharsteinEtal2014.pdf)]

Rainy datasets
* [DID-MDN](https://github.com/hezhangsprinter/DID-MDN) [[Density-aware Single Image De-raining using a Multi-stream Dense Network
](https://arxiv.org/abs/1802.07412)]
#### Train

 ```shell
 python main.py --net='mstn' --crop --crop_size=240 --bs=16 --lr=0.0001 --trainset='its_train' --testset='its_test' --steps=50000000 --eval_step=5000
 ```

#### Test

Trained_models are available at baidudrive: https://drive.google.com/drive/folders/1U7zq8AyZFivzMg_yuaPCbIa5rF0wshdE?usp=sharing

 ```shell
 python runtest.py --test_imgs=datapath --ck='pre_models' --dir='savepath'
```
