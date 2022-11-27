import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from architectures import resnet34
import torch
from torch import nn


if __name__ == '__main__':
    use_ema = True
    device = torch.device('cuda')
    # parameterss
    img_mean = 292.483 # mean/std of image intensity, if it is None, then we use the mean/std of the inputs
    img_std = 321.499 
    
    img_good = np.load('../examples/good.npy')
    img_bad = np.load('../examples/bad.npy')

    img = np.stack((img_good, img_bad),axis=0)
    
    # load model
    model = resnet34(pretrained=False, num_classes=3)
    model = nn.DataParallel(model).to(device)
    checkpoint = torch.load('pretrained_models/pytorch.ckpt', map_location=device)
    state_dict = checkpoint['ema_state_dict' if use_ema else 'state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # the size of image should be 256x256
    assert img.shape[1] == 256 and img.shape[2] == 256
    
    # standardization
    if img_mean is None:
        img_mean = np.mean(img)
    if img_std is None:
        img_std = np.std(img)
    img = (img - img_mean) / img_std
    
    # the input to the network should have 3 channels
    img = np.stack((img, img, img),axis=1)
    
    # predict
    with torch.no_grad():
        pred, _ = model(torch.tensor(img, dtype=torch.float32, device=device))
        pred = torch.softmax(pred, dim=1).cpu().numpy()
        
    # for the miccai model, the ouptut would be 3 values for each image, i.e., [out of ROI, high-quality, low-quality]
    print('                   prob. out-of-ROI       prob. high-quality       prob. low-quality')
    print('example_good               %1.6f                 %1.6f                %1.6f' % (pred[0,0],pred[0,1], pred[0,2]))
    print(' example_bad               %1.6f                 %1.6f                %1.6f' % (pred[1,0],pred[1,1], pred[1,2]))
