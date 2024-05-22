# import openslide

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'D:\openslide\openslide-bin-4.0.0.2-windows-x64\openslide-bin-4.0.0.2-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        print("Imported openslide - from new path")
else:
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
import glob
#
wsi_path = r'D:\Data\tls-data\selected_raw_data'
xml_path = r'D:\Data\tls-data\selected_raw_data'
target_dir = r'D:\Data\tls-data\selected_patched-data'
# Check if the directory exists
if not os.path.exists(target_dir):
    # If not, create the directory
    os.makedirs(target_dir)
     

# List all the .svs files in the directory
wsi_files = glob.glob(wsi_path + '/*.svs')
n_files = len(wsi_files)
print(f"Total number of svs files to process: {n_files}")

# Check if the corresponding .xml files exist for each .svs file
# for wsi in wsi_files:
#     # Get the base name of the file (without extension)
#     base_name = os.path.splitext(wsi)[0]
#     # Construct the path to the corresponding .xml file
#     anno_path = os.path.join(xml_path, base_name + '.xml')
#     # Check if the .xml file exists
#     if os.path.exists(anno_path):
#         print(f"Annotation file for {wsi} exists.")
#     else:
#         print(f"Annotation file for {wsi} does not exist.")

# Check if the corresponding .xml files exist for each .svs file
wsi_files = [wsi for wsi in wsi_files if os.path.exists(os.path.join(xml_path, os.path.splitext(wsi)[0] + '.xml'))]

print(f"Total number of svs files with corresponding xml files: {len(wsi_files)}")

wsi_counter=0
for wsi in wsi_files:
    wsi_counter+=1
    print(f"Preprocessing slide number: {wsi_counter} - {wsi}")
    slide = openslide.open_slide(wsi)
    # pdb.set_trace()
    # if (wsi[-3:] == 'tif'):
    processed_tiles = 0  # Counter for processed tiles
    not_processed_tiles = 0  # Counter for not processed tiles

    anno_path = os.path.join(xml_path, os.path.splitext(wsi)[0] + '.xml')

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
    # plt.imsave('masked_wsi.jpg',masked_image)
    # plt.imsave('mask.jpg',mask, cmap="gray")
    print("Step 2/2: Converting to patches")
    #Generate object for tiles using the DeepZoomGenerator
    tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
    tiles_gt = create_gt_tiles(mask, 256)
    #Here, we have divided our svs into tiles of size 256 with no overlap. 
    
    wsi_name_ts = target_dir+'/'+ 'tissue_patches/'
    wsi_name_gt = target_dir+'/'+ 'annotations/'
    wsi_name_bg = target_dir+'/'+'backgroud/'
    os.makedirs(wsi_name_ts, exist_ok=True)
    os.makedirs(wsi_name_gt, exist_ok=True)
    os.makedirs(wsi_name_bg, exist_ok=True)
    
    # # Ensure the directory exists
    # os.makedirs(os.path.dirname(wsi_name), exist_ok=True)

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
            base_name = os.path.basename(wsi)  # New line
            base_name_no_ext = os.path.splitext(base_name)[0]  # New line
            

            if (temp_tile_np.mean() < 235 and temp_tile_np.std() > 15):

                print("Processing tile number:", tile_name)
                #tifffile giving weird bugs so i use imageio
                # pdb.set_trace()
                #
                save_file_name = os.path.join(wsi_name_ts, base_name_no_ext + tile_name + ".tif")  # Corrected line
                imageio.imwrite(save_file_name, temp_tile_np)
                # pdb.set_trace()
                #plt.imsave(wsi_name_gt + wsi.replace(wsi[-4:],'') + tile_name + ".jpg",tiles_gt[tile_count][2],cmap="gray")
                #ann_file_name = os.path.join(wsi_name_gt, base_name_no_ext + tile_name + ".tif")  # Corrected line
                save_file_name = os.path.join(wsi_name_gt, base_name_no_ext + tile_name + ".tif")  # Corrected line
                imageio.imwrite(save_file_name, (tiles_gt[tile_count][2]*255).astype(np.uint8))
                processed_tiles += 1  # Increment processed tiles counter
            else:
                print("NOT PROCESSING TILE:", tile_name)
                # pdb.set_trace()
                #bg_file_name = os.path.join(wsi_name_bg, tile_name + ".tif")  # Corrected line
                save_file_name = os.path.join(wsi_name_bg, base_name_no_ext + tile_name + ".tif")  # Corrected line
                tiff.imwrite(save_file_name, temp_tile_np)
                not_processed_tiles += 1  # Increment not processed tiles counter
            tile_count+=1
    # print(f"Step 3/3: Making GT patches")
    print(f"***Processed Slide-{wsi_counter}***")
    print(f"Number of tiles: {tile_count}")
    print(f"Number of processed tiles: {processed_tiles}")
    print(f"Number of not processed tiles: {not_processed_tiles}")
    # pdb.set_trace()
