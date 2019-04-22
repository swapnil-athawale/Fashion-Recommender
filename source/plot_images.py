# -*- coding: utf-8 -*-


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import glob
from PIL import Image
import numpy as np
#input_shape=(200,200,3)

import matplotlib.pyplot as plt
CLASSES = 46



X = []
labels = []
path='/DATA/athawale.1/fashion_attribute/train/'
os.chdir(path)
director = sorted(glob.glob('*'))
count=0
for i in range(CLASSES):
	os.chdir(path)
	os.chdir(director[i])
	directories = sorted(glob.glob('*'))
	os.chdir(directories[0])
	#print(directories[0])
	imgs = sorted(glob.glob('*.jpg'))
	#print(len(imgs))
	img = Image.open(path+director[i]+'/'+directories[0]+'/'+imgs[0])
	X.append(img)
	labels.append(director[i]) 



# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,8), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

os.chdir(path)
#save_img = plots(X, titles=labels, rows=4)
#save_img = plots(X[:23], titles=labels[:23], rows=4)
save_img1 = plots(X[23:], titles=labels[23:], rows=4)

plt.savefig('part2.jpg')























