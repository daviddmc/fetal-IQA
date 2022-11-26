# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import torchvision.transforms.functional as F


LOG = logging.getLogger('main')
NO_LABEL = -1




class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
        

class HASTETransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        if isinstance(out1, tuple):
            out1, out3 = out1
            #rr = np.random.rand()
            #if rr < 0.05:
            #    Image.fromarray(np.squeeze(255*(out3.numpy()*0.1993 + 0.1879)).astype(np.uint8)).save('%f.png' % rr)
            out2 = self.transform(inp[0])
        else:
            out3 = 0
            out2 = self.transform(inp)
        return out1, out2, out3
        
'''
class HASTETransformOnece:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        return self.transform(inp[:, :, :1])
'''

def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
    

class TrainTransform:
    def __init__(self):
        pass
        
    def __call__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        #transforms.ToPILImage(),
        args = tuple(Image.fromarray(img) for img in args)
        #transforms.RandomRotation(10),
        angle = np.random.uniform(-10, 10)
        args = tuple(img.rotate(angle) for img in args)
        #transforms.RandomCrop(256, pad_if_needed=True),
        if args[0].size[0] < 256:
            args = tuple(F.pad(img, (256 - img.size[0], 0), 0, 'constant') for img in args)
        if args[0].size[1] < 256:
            args = tuple(F.pad(img, (0, 256 - img.size[1]), 0, 'constant') for img in args)
        i, j, h, w = self.get_params(args[0])
        args = tuple(F.crop(img, i, j, h, w) for img in args)
        #transforms.RandomHorizontalFlip(),
        if np.random.random() < 0.5:
            args = tuple(F.hflip(img) for img in args)
        #transforms.RandomVerticalFlip(),
        if np.random.random() < 0.5:
            args = tuple(F.vflip(img) for img in args)
        #transforms.ToTensor(),
        args = tuple(F.to_tensor(img) for img in args)
        #transforms.Normalize(**channel_stats, inplace=True)
        args = tuple(F.normalize(tensor, [292.483], [321.499], True) for tensor in args)
        if len(args) == 1:
            return args[0]
        else:
            return args
            
    def get_params(self, img):
        w, h = img.size
        th, tw = 256, 256
        i = 0 if h == th else np.random.randint(0, h - th)
        j = 0 if w == tw else np.random.randint(0, w - tw)
        return i, j, th, tw
        
        
