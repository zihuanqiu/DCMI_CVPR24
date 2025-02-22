# DCMI CVPR24

This repository contains the PyTorch implementation of the paper "Dual-consistency Model Inversion for Non-exemplar Class Incremental Learning" (CVPR 2024).

Paper Link: (https://openaccess.thecvf.com/content/CVPR2024/html/Qiu_Dual-Consistency_Model_Inversion_for_Non-Exemplar_Class_Incremental_Learning_CVPR_2024_paper.html)


## Overview
This work introduces a novel approach for non-exemplar class incremental learning, leveraging dual-consistency model inversion to synthesize old task samples
to improve performance.


## Usage

### Training on CIFAR-100 (5-task)
To train the model on CIFAR-100 with 5 incremental tasks from scratch, run:
```
python main.py --gpu='0' --start_task=0 --task_num=5 --fg_nc=50
```

### Training on Tiny-ImageNet (5-task)
To train the model on Tiny-ImageNet with 5 incremental tasks from scratch, navigate to the Tiny-ImageNet directory and run:
```
cd Tiny-ImageNet
python main_tiny.py --gpu='0' --start_task=0 --task_num=5 --fg_nc=100
```

### Parameters
- --gpu: Specify the GPU ID to use.
- --start_task: The task number to start training from (useful for resuming).
- --task_num: Total number of incremental tasks.
- --fg_nc: Number of classes in the initial task.


## Acknowledgment
This project builds upon the code framework from CVPR21_PASS (https://github.com/Impression2805/CVPR21_PASS). We thank the authors for their valuable contribution.


## Citation
If you find this work useful, please cite our paper:
```
@InProceedings{Qiu_2024_CVPR,
    author    = {Qiu, Zihuan and Xu, Yi and Meng, Fanman and Li, Hongliang and Xu, Linfeng and Wu, Qingbo},
    title     = {Dual-Consistency Model Inversion for Non-Exemplar Class Incremental Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24025-24035}
}
```
