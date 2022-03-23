import numpy as np
import os
import sys
sys.path.append('.')
from mask_generation.utils import MaskGeneration, MergeMask
import glob
from PIL import Image
import random
from multiprocessing import Pool
import time

direction = ['horizontal', 'vertical']

def get_mask(img_path, out_dir):
    array = np.array(Image.open(img_path).resize((512,512), Image.BICUBIC))
    index = random.randint(0,6)
    print('mode index:', index)
    if index == 0:
        mode = {
            'name': 'ThickStrokes',
            'size': 512,
        }
    elif index == 1:
        mode = {
            'name': 'MediumStrokes',
            'size': 512,
        }
    elif index == 2:
        mode = {
            'name': 'ThinStrokes',
            'size': 512,
        }
    elif index == 3:
        mode = {
            'name': 'Every_N_Lines',
            'n': random.randint(2, 8),
            'direction': random.choice(direction)
        }
    elif index == 4:
        mode = {
            'name': 'Completion',
            'ratio': random.uniform(0.05, 1),
            'direction': random.choice(direction),
            'reverse': False,
        }
    elif index == 5:
        mode = {
            'name': 'Expand',
            'size': 'random', # None means half of size
            #'direction': random.choice(['interior', 'exterior'])
            'direction': 'interior'
        }
    elif index == 6:
        mode = {
            'name': 'Nearest_Neighbor',
            'scale': random.randint(2, 6),
            'upsampling': False,
        }
    else:
        raise NameError(index)

    mask_generation = MaskGeneration()
    gt, mask = mask_generation(array, mode, verbose=False)
    # out = np.concatenate((gt, mask), axis=1)
    # mask -> 255: inpaint, 0: keep
    # out = MergeMask(array, 255 - mask)
    name = os.path.basename(img_path)
    Image.fromarray(mask).save(os.path.join(out_dir, name))

if __name__ == '__main__':
    start = time.time()
    random.seed(1)
    np.random.seed(1)
    out_dir = '/mnt/bd/aurora-mtrc-sxz/datas/FFHQ/train/inpainting_mask'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image_dir = glob.glob('/mnt/bd/aurora-mtrc-sxz/datas/FFHQ/train/images/*.png')
    print('lenght: ', len(image_dir))
    pool = Pool()
    for item in image_dir:
        pool.apply_async(get_mask, (item, out_dir))
    pool.close()
    pool.join()
    print('use time: ', time.time() - start)
