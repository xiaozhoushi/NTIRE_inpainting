import numpy as np
import os
import sys
sys.path.append('.')
from mask_generation.utils import MaskGeneration, MergeMask
import glob
from PIL import Image
import random
from multiprocessing import Pool

def get_mask(img_path, out_dir):
    array = np.array(Image.open(img_path))
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
            'n': 2,
            'direction': 'horizontal'
        }
    elif index == 4:
        mode = {
            'name': 'Completion',
            'ratio': 0.5,
            'direction': 'horizontal',
            'reverse': False,
        }
    elif index == 5:
        mode = {
            'name': 'Expand',
            'size': None, # None means half of size
            'direction': 'interior'
        }
    elif index == 6:
        mode = {
            'name': 'Nearest_Neighbor',
            'scale': 4,
            'upsampling': False,
        }
    else:
        raise NameError(index)

    mask_generation = MaskGeneration()
    gt, mask = mask_generation(array, mode, verbose=True)
    # out = np.concatenate((gt, mask), axis=1)
    # mask -> 255: inpaint, 0: keep
    # out = MergeMask(array, 255 - mask)
    name = os.path.basename(img_path).split('.')[0] + '_mask.png'
    Image.fromarray(mask).save(os.path.join(out_dir, name))

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    out_dir = '/mnt/bd/aurora-mtrc-sxz/datas/FFHQ/train_inpaint_mask'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image_dir = glob.glob('/mnt/bd/aurora-mtrc-sxz/datas/FFHQ/train/*.png')
    pool = Pool()
    for item in image_dir:
        pool.apply_async(get_mask,[item, out_dir])
    pool.close()
    pool.join()
    print('end process')
