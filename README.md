# SelfTraining-6D

This repo provides for the implementation of the ECCV'22 paper:

#### Sim-to-Real 6D Object Pose Estimation via Iterative Self-training for Robotic Bin Picking [arXiv](https://arxiv.org/pdf/2204.07049.pdf)

## Requirement

- Ubuntu 18.04, CUDA 10.2, Python >= 3.6
- kaolin == 0.1.0
- opencv-python == 4.5.4.58

## Installation

Compile the knn module:
```bash
cd lib/knn
python setup.py install --user
```

Compile the ransac voting layer:
```bash
cd lib/ransac_voting
python setup.py install --user
```

Install kaolin
```bash
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
git checkout v0.1
python setup.py develop
```
## Dataset
Download our processed ROBI dataset from [here] and put them into 'SelfTraining-6D/data'

## Virtual Training
Following [object-posenet](https://github.com/mentian/object-posenet) to train an object pose estimation model on our provided virtual data. Put the virtual model into 'SelfTraining-6D/virtual_models'. To skip this step, you can download our provided virtual model from [here].

## Sim-to-Real Training
self_training.py is the main file for sim-to-real self-training.

Example:
```bash
python self_training.py --dataset zigzag --nepoch 30 --iter 10
```
The intermediate data with pseudo labels will be stored into 'SelfTraining-6D/data'. The trained model will be stored into 'SelfTraining-6D/real_models'

## Evaluation
Example:
```bash
python evaluate.py --obj_name zigzag --testing_mode st --testing_iter 5
```



## Citation
If you find this repo helpful, please consider citing:
```latex
@InProceedings{chen_2022_sim,
  title     = {Sim-to-Real 6D Object Pose Estimation via Iterative Self-training for Robotic Bin Picking},
  author    = {Chen, Kai and Cao, Rui and James, Stephen and Li, Yichuan and Liu, Yun-Hui and Abbeel, Pieter and Dou, Qi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  month     = {October},
  year      = {2022}
}
```
Any questions, please feel free to contact Kai Chen (kaichen@cse.cuhk.edu.hk).