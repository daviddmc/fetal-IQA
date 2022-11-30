# Image quality assessment for fetal MRI
This repo is the implementation of an image quality assessment (IQA) method for fetal MRI, which is the accumulation of the following works:

\[1\] Semi-supervised learning for fetal brain MRI quality assessment with ROI consistency ([MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_37) | [arXiv](https://arxiv.org/abs/2006.12704))

\[2\] Automated detection and reacquisition of motion-degraded images in fetal HASTE imaging at 3 T ([MRM](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29106))

\[3\] A deep learning approach for image quality assessment of fetal brain MRI ([ISMRM](https://archive.ismrm.org/2019/0839.html))

## Usage

### Train your own models

#### Brain segmentation (optional)

To use ROI consistency, you would need to generate ROI for your dataset.

1. Download the [pre-trained segmentation network](https://bitbucket.org/bchradiology/u-net/src/master/Model/)
2. Modifty `PATH_LABELED_DATA` and `PATH_UNLABELED_DATA` in `brainSeg/Code/FetalUnet.py` to point to your own dataset.
3. Run:
    ```
    cd brainSeg/Code
    python FetalUnet.py
    ```

#### Implement your dataset

Implement your dataset following `src/mean_teacher/haste.py`

#### Training

```
cd src
python experiments/haste_exp.py
```

### Use pre-trained model

#### PyTorch

1. Download [pre-trained models](https://zenodo.org/record/7368570) (`pytorch.ckpt`) to `torch_iqa_tool/pretrained_models`

2. run demo
    ```
    cd torch_iqa_tool
    python iqa_demo.py
    ```

#### Tensorflow

1. Download [pre-trained models](https://zenodo.org/record/7368570) (`model_ismrm.hdf5` and `model_miccai.h5`) to `tf_iqa_tool/pretrained_models`

2. run demo
    ```
    cd tf_iqa_tool
    python iqa_demo.py
    ```

## Cite our work
```
@inproceedings{xu2020semi,
  title={Semi-supervised learning for fetal brain MRI quality assessment with ROI consistency},
  author={Xu, Junshen and Lala, Sayeri and Gagoski, Borjan and Abaci Turk, Esra and Grant, P Ellen and Golland, Polina and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={386--395},
  year={2020},
  organization={Springer}
}

@article{gagoski2022automated,
  title={Automated detection and reacquisition of motion-degraded images in fetal HASTE imaging at 3 T},
  author={Gagoski, Borjan and Xu, Junshen and Wighton, Paul and Tisdall, M Dylan and Frost, Robert and Lo, Wei-Ching and Golland, Polina and van Der Kouwe, Andre and Adalsteinsson, Elfar and Grant, P Ellen},
  journal={Magnetic Resonance in Medicine},
  volume={87},
  number={4},
  pages={1914--1922},
  year={2022},
  publisher={Wiley Online Library}
}

@inproceedings{lala2019deep,
  title={A deep learning approach for image quality assessment of fetal brain MRI},
  author={Lala, Sayeri and Singh, Nalini and Gagoski, Borjan and Turk, Esra and Grant, P Ellen and Golland, Polina and Adalsteinsson, Elfar}
  booktitle={Proceedings of the International Society for Magnetic Resonance in Medicine},
  year={2019},
}
```
