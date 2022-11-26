import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pydicom
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

import cv2
import pickle

class Segmentation:
    def __init__(self):
        
        # Network Parameters
        tf.reset_default_graph()
        width = 256
        height = 256
        n_channels = 1
        n_classes = 2 # total classes (brain, non-brain)
        x = tf.placeholder(tf.float32, [None, height, width, n_channels])
        self.x = x
        
        ################Create Model######################
        conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
        pool1 = max_pool_2d(conv1, 2)
        
        conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
        pool2 = max_pool_2d(conv2, 2)
        
        conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
        pool3 = max_pool_2d(conv3, 2)
        
        conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
        pool4 = max_pool_2d(conv4, 2)
        
        conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
        conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")
        
        up6 = upsample_2d(conv5,2)
        up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
        conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
        conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")
        
        up7 = upsample_2d(conv6,2)
        up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
        conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
        conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")
        
        up8 = upsample_2d(conv7,2)
        up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
        conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
        conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")
        
        up9 = upsample_2d(conv8,2)
        up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
        conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
        conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")
        
        self.pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')
        
        #self.pred = tf.reshape(pred, [-1, n_classes])
        
        ###############Initialize Model#######################
        init = tf.initialize_all_variables()
            
        self.sess = tf.Session()
        self.sess.run(init)
        
        saver = tf.train.Saver()
        #Load model
        model_path = "../Model/Fetal_2D_Ref_6980_norm0.ckpt"
        saver.restore(self.sess, model_path)
        
    def segment(self, image_data): # N x H x W
        t = time.time()
        if image_data.shape[1]>=256:
            image_data = image_data[:, (image_data.shape[1]-256)//2:(image_data.shape[1]-256)//2+256, :]
        else:
            image_data = np.pad(image_data, ((0,0), (0, 256-image_data.shape[1]), (0, 0)))
        if image_data.shape[2]>=256:
            image_data = image_data[:, :, (image_data.shape[2]-256)//2:(image_data.shape[2]-256)//2+256]
        else:
            image_data = np.pad(image_data, ((0,0), (0,0), (0, 256-image_data.shape[2])))
        input_data = image_data[..., np.newaxis] # Add one axis to the end
        out = self.sess.run(self.pred, feed_dict={self.x: input_data}) # Find probabilities
        mask = 1 - np.argmax(np.asarray(out), axis=3)
        elapsed = time.time() - t 
        print(elapsed)
        return mask.astype(np.uint8)
        
    def bounding_box(self, masks, images, name):
    
        A_th = 900
    
        A = np.zeros(len(masks))
        C = np.zeros((len(masks), 2))
        R = np.zeros(len(masks))
        for i, mask in enumerate(masks):
            n_label, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)
            if n_label > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                R[i] = np.sqrt(stats[largest_label, cv2.CC_STAT_HEIGHT]**2 + stats[largest_label, cv2.CC_STAT_WIDTH]**2) / 2
                C[i] = centroids[largest_label]
                A[i] = stats[largest_label, cv2.CC_STAT_AREA]
                #print(centroids[largest_label], stats[largest_label, cv2.CC_STAT_AREA])
        
        large_idx = A > A_th
        
        if np.sum(large_idx) >= 2:
            dy = (images.shape[1]-256)//2 if images.shape[1] >= 256 else 0
            dx = (images.shape[2]-256)//2 if images.shape[2] >= 256 else 0
            c = np.average(C[large_idx], axis=0, weights=A[large_idx])
            cx, cy = int(c[0]) + dx, int(c[1]) + dy
            std = int(np.sqrt(np.sum(np.average((C[large_idx] - c)**2, axis=0, weights=A[large_idx]))))
            r = int(R[np.argmax(A)])
        else:
            cx, cy, r, std = 0, 0, 0, 0
        return cx, cy, r, std
            
                
                
def read_dicoms(folder):
    files = list(f for f in sorted(os.listdir(folder)) if f.endswith('.dcm'))
    for f in files:
        if not (50 * 1024 <= os.path.getsize(os.path.join(folder, f)) <= 400 * 1024):
            return None
    if len(files):
        return np.stack(list(pydicom.dcmread(os.path.join(folder, f)).pixel_array.astype(np.float) for f in files))
    else:
        return None
    
                
if __name__ == '__main__':

    is_unlabeled = True
    seg = Segmentation()

    PATH_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
    PATH_LABELED_DATA = os.path.join(PATH_ROOT, 'iqa_data_source_rep/reorganized_filtered_singleton')
    PATH_UNLABELED_DATA = os.path.join(PATH_ROOT, 'iqa_data_source_rep/FetalHASTEReposatoryFromBCH_Anon/')

    f = PATH_UNLABELED_DATA if is_unlabeled else PATH_LABELED_DATA
    
    res_dict = dict()
    
    if is_unlabeled:
        for subject in os.listdir(f):
            for i, sub in enumerate(os.listdir(os.path.join(f, subject))):
                for j, stack in enumerate(os.listdir(os.path.join(f, subject, sub))):
                    if not stack.startswith('T2_HASTE'):
                        continue
                    name = '-'.join([subject, str(i), str(j)])
                    print(name)
                    imgs = read_dicoms(os.path.join(f, subject, sub, stack))
                    if imgs is None:
                        continue
                    masks = seg.segment(imgs)
                    res = seg.bounding_box(masks, imgs, name)
                    res_dict[(subject, sub, stack)] = res
    else:
        for subject in os.listdir(f):
            for j, stack in enumerate(os.listdir(os.path.join(f, subject))):
                if stack.startswith('.'):
                    continue
                name = '-'.join([subject, str(j)])
                print(name)
                imgs = read_dicoms(os.path.join(f, subject, stack, 'dicoms'))
                if imgs is None:
                    continue
                masks = seg.segment(imgs)
                res = seg.bounding_box(masks, imgs, name)
                res_dict[(subject, stack)] = res
    with open('unlabeled.obj' if is_unlabeled else 'labeled.obj', 'wb') as filehandler:
        pickle.dump(res_dict, filehandler)
    
        
