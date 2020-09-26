# L-CNN — End-to-End Wireframe Parsing

This repository contains the official PyTorch implementation of the paper:  *[Yichao Zhou](https://yichaozhou.com), [Haozhi Qi](http://haozhi.io), [Yi Ma](https://people.eecs.berkeley.edu/~yima/). ["End-to-End Wireframe Parsing."](https://arxiv.org/abs/1905.03246)  ICCV 2019*.

## Introduction

[L-CNN](https://arxiv.org/abs/1905.03246) is a conceptually simple yet effective neural network for detecting the wireframe from a given image. It outperforms the previous state-of-the-art wireframe and line detectors by a large margin. We hope that this repository serves as an easily reproducible baseline for future researches in this area.

## Main Results

### Qualitative Measures

|     <img src="figs/000452_LSD.png" width="180">      |    <img src="figs/000452_AFM.png" width="180">    |     <img src="figs/000452_WF.png" width="180">      | <img src="figs/000452_LCNN.png" width="180"> | <img src="figs/000452_GT.png" width="180"> |
| :--------------------------------------------------: | :-----------------------------------------------: | :-------------------------------------------------: | :------------------------------------------: | :----------------------------------------: |
| [LSD](https://ieeexplore.ieee.org/document/4731268/) | [AFM](https://github.com/cherubicXN/afm_cvpr2019) | [Wireframe](https://github.com/huangkuns/wireframe) |                  **L-CNN**                   |                Ground Truth                |

More random sampled results can be found in the [supplementary material](https://yichaozhou.com/publication/1904lcnn/appendix.pdf) of the paper.

### Quantitative Measures

The following table reports the performance metrics of several wireframe and line detectors on the [ShanghaiTech dataset](https://github.com/huangkuns/wireframe).

|                                                      | ShanghaiTech (sAP<sup>10</sup>) | ShanghaiTech (AP<sup>H</sup>) | ShanghaiTech (F<sup>H</sup>) | ShanghaiTech (mAP<sup>J</sup>) | 
| :--------------------------------------------------: | :--------------------------------: | :-----------------------------: | :----------------------------: | :------------------------------: | 
| [LSD](https://ieeexplore.ieee.org/document/4731268/) |                 /                  |              52.0             |              61.0                |                /                 |
|  [AFM](https://github.com/cherubicXN/afm_cvpr2019)   |                24.4                |              69.5               |              77.2              |               23.3               |
| [Wireframe](https://github.com/huangkuns/wireframe)  |                5.1                 |              67.8               |              72.6              |               40.9               |
|                      **L-CNN**                       |              **62.9**              |            **82.8**             |            **81.2**            |             **59.3**             |

### Precision-Recall Curves
<p align="center">
<img src="figs/PR-APH.svg"  width="400">
<img src="figs/PR-sAP10.svg" width="400">
</p>

## Code Structure

Below is a quick overview of the function of each file.

```bash
########################### Data ###########################
figs/
data/                           # default folder for placing the data
    wireframe/                  # folder for ShanghaiTech dataset (Huang et al.)
logs/                           # default folder for storing the output during training
########################### Code ###########################
config/                         # neural network hyper-parameters and configurations
    wireframe.yaml              # default parameter for ShanghaiTech dataset
dataset/                        # all scripts related to data generation
    wireframe.py                # script for pre-processing the ShanghaiTech dataset to npz
misc/                           # misc scripts that are not important
    draw-wireframe.py           # script for generating figure grids
    lsd.py                      # script for generating npz files for LSD
    plot-sAP.py                 # script for plotting sAP10 for all algorithms
lcnn/                           # lcnn module so you can "import lcnn" in other scripts
    models/                     # neural network structure
        hourglass_pose.py       # backbone network (stacked hourglass)
        line_vectorizer.py      # sampler and line verification network
        multitask_learner.py    # network for multi-task learning
    datasets.py                 # reading the training data
    metrics.py                  # functions for evaluation metrics
    trainer.py                  # trainer
    config.py                   # global variables for configuration
    utils.py                    # misc functions
demo.py                         # script for detecting wireframes for an image
eval-sAP.py                     # script for sAP evaluation
eval-APH.py                     # script for APH evaluation
eval-mAPJ.py                    # script for mAPJ evaluation
train.py                        # script for training the neural network
post.py                         # script for post-processing
process.py                      # script for processing a dataset from a checkpoint
```

## Reproducing Results

### Installation

For the ease of reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before following executing the following commands. 

```bash
git clone https://github.com/zhou13/lcnn
cd lcnn
conda create -y -n lcnn
source activate lcnn
# Replace cudatoolkit=10.1 with your CUDA version: https://pytorch.org/
conda install -y pytorch cudatoolkit=10.1 -c pytorch
conda install -y tensorboardx -c conda-forge
conda install -y pyyaml docopt matplotlib scikit-image opencv
mkdir data logs post
```

### Pre-trained Models

You can download our reference pre-trained models from [Google
Drive](https://drive.google.com/file/d/1NvZkEqWNUBAfuhFPNGiCItjy4iU0UOy2).  Those models were
trained with `config/wireframe.yaml` for 312k iterations.  Use `demo.py`, `process.py`, and
`eval-*.py` to evaluate the pre-trained models.

### Detect Wireframes for Your Own Images
To test LCNN on your own images, you need download the pre-trained models and execute

```Bash
python ./demo.py -d 0 config/wireframe.yaml <path-to-pretrained-pth> <path-to-image>
```
Here, `-d 0` is specifying the GPU ID used for evaluation, and you can specify `-d ""` to force CPU inference.


### Downloading the Processed Dataset
Make sure `curl` is installed on your system and execute
```bash
cd data
../misc/gdrive-download.sh 1T4_6Nb5r4yAXre3lf-zpmp3RbmyP1t9q wireframe.tar.xz
tar xf wireframe.tar.xz
rm wireframe.tar.xz
cd ..
```

If `gdrive-download.sh` does not work for you, you can download the pre-processed dataset
`wireframe.tar.xz` manually from [Google
Drive](https://drive.google.com/drive/u/1/folders/1rXLAh5VIj8jwf8vLfuZncStihRO2chFr) and proceed
accordingly.

#### Processing the Dataset
*Optionally*, you can pre-process (e.g., generate heat maps, do data augmentation) the dataset from
scratch rather than downloading the processed one.  **Skip** this section if you just want to use
the pre-processed dataset `wireframe.tar.xz`.

```bash
cd data
../misc/gdrive-download.sh 1BRkqyi5CKPQF6IYzj_dQxZFQl0OwbzOf wireframe_raw.tar.xz
tar xf wireframe_raw.tar.xz
rm wireframe_raw.tar.xz
cd ..
dataset/wireframe.py data/wireframe_raw data/wireframe
```

### Training
The default batch size assumes your have a graphics card with 12GB video memory, e.g., GTX 1080Ti or RTX 2080Ti. You may reduce the batch size if you have less video memory.

To train the neural network on GPU 0 (specified by `-d 0`) with the default parameters, execute
```bash
python ./train.py -d 0 --identifier baseline config/wireframe.yaml
```

## Testing Pretrained Models
To generate wireframes on the validation dataset with the pretrained model, execute

```bash
./process.py config/wireframe.yaml <path-to-checkpoint.pth> data/wireframe logs/pretrained-model/npz/000312000
```

### Post Processing

To post process the outputs from neural network (only necessary if you are going to evaluate AP<sup>H</sup>), execute
```bash
python ./post.py --plot --thresholds="0.010,0.015" logs/RUN/npz/ITERATION post/RUN-ITERATION
```
where ``--plot`` is an *optional* argument to control whether the program should also generate
images for visualization in addition to the npz files that contain the line information, and
``--thresholds`` controls how aggressive the post processing is.  Multiple values in ``--thresholds``
is convenient for hyper-parameter search.  You should  replace `RUN` and `ITERATION` to the
desired value of your training instance.

### Evaluation

To evaluate the sAP (recommended) of all your checkpoints under `logs/`, execute
```bash
python eval-sAP.py logs/*/npz/*
```

To evaluate the mAP<sup>J</sup>, execute
```bash
python eval-mAPJ.py logs/*/npz/*
```

To evaluate AP<sup>H</sup>, you first need to post process your result (see the previous section).
In addition, **MATLAB is required for AP<sup>H</sup> evaluation** and `matlab` should be under your
`$PATH`.  The **parallel computing toolbox** is highly suggested due to the usage of `parfor`.
After post processing, execute

```bash
python eval-APH.py post/RUN-ITERATION/0_010 post/RUN-ITERATION/0_010-APH
```
to get the plot, where `0_010` is the threshold used in the post processing, and `post/RUN-ITERATION-APH`
is the temporary directory storing intermediate files.  Due to the usage of pixel-wise matching, 
the evaluation of AP<sup>H</sup> **may take up to an hour** depending on your CPUs.

See the source code of `eval-sAP.py`, `eval-mAPJ.py`, `eval-APH.py`, and `misc/*.py` for more
details on evaluation.

### Citing End-to-End Wireframe Parsing

If you find L-CNN useful in your research, please consider citing:

```
@inproceedings{zhou2019end,
 author={Zhou, Yichao and Qi, Haozhi and Ma, Yi},
 title={End-to-End Wireframe Parsing},
 booktitle={ICCV 2019},
 year={2019}
}
```
