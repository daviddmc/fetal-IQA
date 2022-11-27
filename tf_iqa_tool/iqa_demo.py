import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow.keras as keras
import numpy as np
import pydicom


if __name__ == '__main__':
    # parameters
    use_miccai_model = False  # use ismrm or miccai models
    img_mean = 292.483 # mean/std of image intensity, if it is None, then we use the mean/std of the inputs
    img_std = 321.499 
    
    # read data for demo
    # img_good = pydicom.dcmread('../examples/good.dcm').pixel_array.astype(np.float32)
    # img_bad = pydicom.dcmread('../examples/bad.dcm').pixel_array.astype(np.float32)
    # np.save('../examples/good.npy', img_good)
    # np.save('../examples/bad.npy', img_bad)

    img_good = np.load('../examples/good.npy')
    img_bad = np.load('../examples/bad.npy')

    img = np.stack((img_good, img_bad),axis=0)
    
    # load model
    if use_miccai_model:
        model_path = './pretrained_models/model_miccai.h5'
    else:
        model_path = './pretrained_models/model_ismrm.hdf5'
    model = keras.models.load_model(model_path, compile=False)
    
    # the size of image should be 256x256
    assert img.shape[1] == 256 and img.shape[2] == 256
    
    # standardization
    if img_mean is None:
        img_mean = np.mean(img)
    if img_std is None:
        img_std = np.std(img)
    img = (img - img_mean) / img_std
    
    # the input to the network should have 3 channels
    img = np.stack((img, img, img),axis=-1)
    
    # predict
    pred = model.predict(x=img)
    if use_miccai_model:
        # for the miccai model, the ouptut would be 3 values for each image, i.e., [out of ROI, high-quality, low-quality]
        print('                   prob. out-of-ROI       prob. high-quality       prob. low-quality')
        print('example_good               %1.6f                 %1.6f                %1.6f' % (pred[0,0],pred[0,1], pred[0,2]))
        print(' example_bad               %1.6f                 %1.6f                %1.6f' % (pred[1,0],pred[1,1], pred[1,2]))
    else:
        # for the ismrm model, the output would be one value for each image, i.e., the probability of being low-quality images
        print('                   prob. low-qaulity')
        print('example_good                %1.6f' % pred[0, 0])
        print(' example_bad                %1.6f' % pred[1, 0])
