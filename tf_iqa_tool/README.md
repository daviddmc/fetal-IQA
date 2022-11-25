
# README

1. See requirements.txt for environment setup.

2. Download [pre-trained models](https://zenodo.org/record/7361788#.Y4DkknbMKUk) to `./pretrained_models`

3. DEMO: run `python iqa_demo.py`.

4. For standardization, please use the mean and std of you own dataset.

5. Both models are trained on data with the size of 256 x 256 and spatial resolution of ~1mm. Try to resize/pad/crop you data if they are not of the same size.

6. For the ismrm model, the output would be one value for each image, i.e., the probability of being low-quality images.

7. For the miccai model, the ouptut would be 3 values for each image, i.e., [out of ROI, high-quality, low-quality].

8. Please try both models and decide which suits you best.
