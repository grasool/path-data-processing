OPENSLIDE_PATH = r'C:\Users\80027294\Documents\openslide-bin-4.0.0.3-windows-x64\openslide-bin-4.0.0.3-windows-x64\bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        print("Imported openslide - from new path")
else:
    import openslide


import cv2
import numpy as np
from utils import norm_HnE
import pdb
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from openslide.deepzoom import DeepZoomGenerator
import tifffile as tiff
from utils import create_gt_tiles,retrieve_ann_vertex,retrieve_ann_coord
import imageio

wsi_path='C:/Users/80027294/Documents/tls-masks/images'
xml_path='C:/Users/80027294/Documents/tls-masks/annotations'
target_dir='C:/Users/80027294/Documents/tls-masks/data_patches'

i=0
for wsi in os.listdir(wsi_path):
    i+=1
    print(f"Preprocessing Slide number {i} i.e {wsi}")
    slide = openslide.open_slide(os.path.join(wsi_path,wsi))
    # pdb.set_trace()
    # if (wsi[-3:] == 'tif'):
    anno_path  = os.path.join(xml_path,wsi.replace(wsi[-3:],''))
    anno_path+='xml'
    tree = ET.parse(anno_path)
    root = tree.getroot()
    # Define the dimensions for the mask (it should match the dimensions of the WSI)
    mask = np.zeros(slide.level_dimensions[0][::-1], dtype=np.uint8)
    # pdb.set_trace()
    if (wsi[-3:]) == 'svs':
        masked_image,mask = retrieve_ann_vertex(slide,root,mask)
    elif (wsi[-3:]=='tif'):
         masked_image,mask = retrieve_ann_coord(slide,root,mask)
    # Display the masked image
    # plt.imshow(masked_image)
    plt.imsave('masked_wsi_gt5.png',masked_image)