import cv2
import os
import tifffile
from matplotlib import pyplot as plt 
  
# create figure 
fig = plt.figure(figsize=(10, 7)) 
# setting values to rows and column variables 
rows = 1
columns = 2
# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1)
data_gt = '/home/afridi/Desktop/moffitt_ali/data_patches/train_gt'
data_patches = '/home/afridi/Desktop/moffitt_ali/data_patches/train_patches'
for gt in os.listdir(data_gt):
    inp = cv2.imread(os.path.join(data_gt,gt))
    if (inp.sum()):
        inp2 = tifffile.imread(os.path.join(data_patches,gt[:-3]+'tif'))
        cv2.imshow('vis',inp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('vis',inp2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()     
        # import pdb; pdb.set_trace()