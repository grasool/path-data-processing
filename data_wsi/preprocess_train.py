import openslide
import os
import cv2
import numpy as np
from normalize_HnE import norm_HnE
import pdb
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from openslide.deepzoom import DeepZoomGenerator
import tifffile as tiff
from utils import create_gt_tiles,retrieve_ann_vertex,retrieve_ann_coord
import imageio
import argparse
#

def main(args):     
    i=0
    wsi_path = args.wsi_path
    xml_path = args.xml_path
    target_dir = args.target_dir
    # wsi_path = '/home/afridi/Desktop/path-data-processing/data_wsi/images/train'
    # xml_path = '/home/afridi/Desktop/path-data-processing/data_wsi/annotations/train'
    # target_dir = '/home/afridi/Desktop/path-data-processing/data_patches'

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
        plt.imsave('masked_wsi_gt.jpg',masked_image)
        # plt.imsave('mask.jpg',mask, cmap="gray")
        print("Step 2/2: Converting to patches")
        #Generate object for tiles using the DeepZoomGenerator
        tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
        tiles_gt = create_gt_tiles(mask, 256)
        #Here, we have divided our svs into tiles of size 256 with no overlap. 
        
        wsi_name = target_dir+'/'+ 'train_patches/'
        wsi_name_gt = target_dir+'/'+ 'train_gt/'
        os.makedirs(wsi_name,exist_ok=True)
        os.makedirs(wsi_name_gt ,exist_ok=True)
        os.makedirs(target_dir+'/np',exist_ok=True)
        
        #pick the high resolution tiles
        max_dim = len(tiles.level_tiles)
        cols, rows = tiles.level_tiles[max_dim-1]   
        tile_count = 0 

        for row in range(rows):
            for col in range(cols):
                tile_name = str(row) + "_" + str(col)
                # my_conv = 
                # pdb.set_trace()
                temp_tile = tiles.get_tile(max_dim-1, (col, row))
                temp_tile_RGB = temp_tile.convert('RGB')
                temp_tile_np = np.array(temp_tile_RGB)
                #Save original tile

                if (temp_tile_np.mean() < 235 and temp_tile_np.std() > 15):

                    print("Processing tile number:", tile_name)
                    #tifffile giving weird bugs so i use imageio
                    # pdb.set_trace()
                    imageio.imwrite(wsi_name + wsi.replace(wsi[-4:],'') + tile_name + ".tif", temp_tile_np)
                    # pdb.set_trace()
                    plt.imsave(wsi_name_gt + wsi.replace(wsi[-4:],'') + tile_name + ".jpg",tiles_gt[tile_count][2],cmap="gray")
                else:
                    print("NOT PROCESSING TILE:", tile_name)
                    # pdb.set_trace()
                    
                    tiff.imwrite(target_dir+'/np' + '/' + tile_name + ".tif", temp_tile_np)
                tile_count+=1
        # print(f"Step 3/3: Making GT patches")
        print(f"***Processed Slide{i}***")

    # pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--wsi_path', type=str)
    parser.add_argument('--xml_path', type=str)
    parser.add_argument('--target_dir', type=str)
    # parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)