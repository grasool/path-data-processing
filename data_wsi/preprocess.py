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
from utils import create_gt_tiles
import imageio

#
wsi_path = '/home/afridi/Desktop/moffitt_ali/data_wsi/images'
xml_path = '/home/afridi/Desktop/moffitt_ali/data_wsi/annotations'
target_dir = '/home/afridi/Desktop/moffitt_ali/data_patches'




i=0
for wsi in os.listdir(wsi_path):
    i+=1
    print(f"Preprocessing Slide number {i}")
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
    # Extract annotations and draw them on the mask
    print("Step 1/2: Retrieving Annotations")
    for annotation in root.findall(".//Annotation"):
            # pdb.set_trace()
        # for region in annotation.findall(".//Region"):
            coordinates = []
            if(annotation.get("Type") == 'Polygon'):
                for vertex in annotation.findall(".//Coordinate"):
                    x = int(float(vertex.get("X")))
                    y = int(float(vertex.get("Y")))
                    coordinates.append((x, y))

                # Convert coordinates to a numpy array
                coordinates = np.array(coordinates, dtype=np.int32)
                cv2.fillPoly(mask, [coordinates], color=(255))
            else:
                    print('Multipolygon found! skipping..')
    
    print(f"WSI Dimensions: {slide.dimensions}")
    # Apply the mask to the WSI
# Load the image at the lowest resolution for visualization
    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
    thumbnail = np.array(thumbnail)
    # Resize the mask to match the thumbnail's size
    resized_mask = cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    # Apply the mask to the thumbnail image
    masked_image = cv2.bitwise_and(thumbnail, thumbnail, mask=resized_mask)

    # Display the masked image
    # plt.imshow(masked_image)
    plt.imsave('masked_wsi.jpg',masked_image)
    plt.imsave('mask.jpg',mask, cmap="gray")
    print("Step 2/2: Converting to patches")
    #Generate object for tiles using the DeepZoomGenerator
    tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
    tiles_gt = create_gt_tiles(mask, 256)
    #Here, we have divided our svs into tiles of size 256 with no overlap. 
    pdb.set_trace()
    wsi_name = target_dir+'/'+ wsi.replace('.tif','')
    os.makedirs(wsi_name,exist_ok=True)
    os.makedirs(wsi_name + 'gt',exist_ok=True)
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
                imageio.imwrite(wsi_name+ '/' + tile_name + ".tif", temp_tile_np)
                # pdb.set_trace()
                plt.imsave(wsi_name + 'gt'+ '/' + tile_name + ".jpg",tiles_gt[tile_count][2],cmap="gray")
            else:
                print("NOT PROCESSING TILE:", tile_name)
                tiff.imwrite(target_dir+'/np' + '/' + tile_name + ".tif", temp_tile_np)
            tile_count+=1
    # print(f"Step 3/3: Making GT patches")
    print(f"***Processed Slide{i}***")
    # pdb.set_trace()

