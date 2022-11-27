# Usage

1. Download [pre-trained models](https://zenodo.org/record/7368570) (`pytorch.ckpt`) to `./pretrained_models`

2. DEMO: run `python iqa_demo.py`.

3. For standardization, please use the mean and std of you own dataset.

4. The model was trained on data with the size of 256 x 256 and spatial resolution of ~1mm. Try to resize/pad/crop you data if they are not of the same size.

5. The ouptut would be 3 values for each image, i.e., [out of ROI, high-quality, low-quality].
