# LearnableDilationNetwork

Source code for "Learning Dilation Factors for Semantic Segmentation of Street Scenes (GCPR2017)"
We modified the convolution layer in standard Caffe deep learning framework.
This is a demo to train a Deeplab-LargeFOV semantic segmentation model with our learnable dilated convolution networks.

# Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.

# Compilation of Caffe
Please follow the instruction from http://caffe.berkeleyvision.org/installation.html to compile our modified Caffe.
```bash
cd caffe-dynamic-dilation
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, if using Anaconda Python, or if cuDNN is desired)
make all
make test
make runtest
```

# Download initialization caffe
Please run our script to download the initialization caffe model.
```bash
bash download_init_caffemodel.sh
```
You can also download this file from http://liangchiehchen.com/projects/DeepLabv2_vgg.html.

# Training
```bash
bash run.sh
```

# Citation
If our work is useful for you, please consider citing:

@inproceedings{yang_gcpr17,

   title={Learning Dilation Factors for Semantic Segmentation of Street Scenes},
 
   author={Yang He and Margret Keuper and Bernt Schiele and Mario Fritz},
 
   booktitle={39th German Conference on Pattern Recognition (GCPR)},
 
   year={2017}
 
}
