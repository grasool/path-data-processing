
# import numpy as np
# import cv2
# import tifffile as tiff
# import openslide
# def retrieve_ann_vertex(slide,root,mask):
#     # Extract annotations and draw them on the mask
#     print("Step 1/2: Retrieving Annotations")
#     for annotation in root.findall(".//Annotation"):
#         for region in annotation.findall(".//Region"):
#             coordinates = []
#             for vertex in region.findall(".//Vertex"):
#                 x = int(float(vertex.get("X")))
#                 y = int(float(vertex.get("Y")))
#                 coordinates.append((x, y))

#             # Convert coordinates to a numpy array
#             coordinates = np.array(coordinates, dtype=np.int32)
#             cv2.fillPoly(mask, [coordinates], color=(255))
#     print(f"WSI Dimensions: {slide.dimensions}")
#     # import pdb; pdb.set_trace()
#     thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
#     thumbnail = np.array(thumbnail)

#     # Resize the mask to match the thumbnail's size
#     resized_mask = cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST)

#     # Apply the mask to the thumbnail image
#     # import pdb; pdb.set_trace()
#     masked_image = cv2.bitwise_and(thumbnail, thumbnail, mask=resized_mask)
#     return masked_image,mask

# def retrieve_ann_coord(slide,root,mask):
#         # Extract annotations and draw them on the mask
#     print("Step 1/2: Retrieving Annotations")
#     for annotation in root.findall(".//Annotation"):
#             # pdb.set_trace()
#         # for region in annotation.findall(".//Region"):
#             coordinates = []
#             if(annotation.get("Type") == 'Polygon'):
#                 for vertex in annotation.findall(".//Coordinate"):
#                     x = int(float(vertex.get("X")))
#                     y = int(float(vertex.get("Y")))
#                     coordinates.append((x, y))

#                 # Convert coordinates to a numpy array
#                 coordinates = np.array(coordinates, dtype=np.int32)
#                 cv2.fillPoly(mask, [coordinates], color=(255))
#             else:
#                     print('Multipolygon found! skipping..')
    
#     print(f"WSI Dimensions: {slide.dimensions}")
#     # Apply the mask to the WSI
# # Load the image at the lowest resolution for visualization
#     thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
#     thumbnail = np.array(thumbnail)
#     # Resize the mask to match the thumbnail's size
#     resized_mask = cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST)

#     # Apply the mask to the thumbnail image
#     masked_image = cv2.bitwise_and(thumbnail, thumbnail, mask=resized_mask)
#     return masked_image,mask

# def create_gt_tiles(array, tile_size):
#     # Determine the number of rows and columns
#     num_rows = int(np.ceil(array.shape[0] / tile_size))
#     num_cols = int(np.ceil(array.shape[1] / tile_size))
#     # import pdb; pdb.set_trace()
#     # Initialize a list to store the tiles
#     tiles = []
    
#     # Create tiles and store them in the list
#     for row in range(num_rows):
#         for col in range(num_cols):
#             tile = array[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size]
#             tiles.append((row, col, tile))
    
#     return tiles

# def find_mean_std_pixel_value(img_list):
    
#     avg_pixel_value = []
#     stddev_pixel_value= []
#     for file in img_list:
#         image = tiff.imread(file)
#         avg = image.mean()
#         std = image.std()
#         avg_pixel_value.append(avg)
#         stddev_pixel_value.append(std)
        
#     avg_pixel_value = np.array(avg_pixel_value)  
#     stddev_pixel_value=np.array(stddev_pixel_value)
        
#     print("Average pixel value for all images is:", avg_pixel_value.mean())
#     print("Average std dev of pixel value for all images is:", stddev_pixel_value.mean())
    
#     return(avg_pixel_value, stddev_pixel_value)


import numpy as np
import cv2
import tifffile as tiff
import openslide
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from openslide.deepzoom import DeepZoomGenerator
import imageio


def retrieve_ann_vertex(slide, root):
    print("Step 1/2: Retrieving Annotations")
    masks = {}
    for annotation in root.findall(".//Annotation"):
        class_id = int(annotation.get("Id"))  # Use annotation Id as class id
        if class_id not in masks:
            masks[class_id] = np.zeros(slide.level_dimensions[0][::-1], dtype=np.uint8)
        for region in annotation.findall(".//Region"):
            coordinates = []
            for vertex in region.findall(".//Vertex"):
                x = int(float(vertex.get("X")))
                y = int(float(vertex.get("Y")))
                coordinates.append((x, y))
            coordinates = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(masks[class_id], [coordinates], color=255)
    
    print(f"WSI Dimensions: {slide.dimensions}")
    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
    thumbnail = np.array(thumbnail)
    
    resized_masks = [cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST) for mask in masks.values()]
    
    masked_images = [cv2.bitwise_and(thumbnail, thumbnail, mask=mask) for mask in resized_masks]
    return masked_images, list(masks.values())

def retrieve_ann_coord(slide, root):
    print("Step 1/2: Retrieving Annotations")
    masks = {}
    for annotation in root.findall(".//Annotation"):
        class_id = int(annotation.get("Id"))  # Use annotation Id as class id
        if class_id not in masks:
            masks[class_id] = np.zeros(slide.level_dimensions[0][::-1], dtype=np.uint8)
        if annotation.get("Type") == 'Polygon':
            coordinates = []
            for vertex in annotation.findall(".//Coordinate"):
                x = int(float(vertex.get("X")))
                y = int(float(vertex.get("Y")))
                coordinates.append((x, y))
            coordinates = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(masks[class_id], [coordinates], color=255)
        else:
            print('Multipolygon found! skipping..')
    
    print(f"WSI Dimensions: {slide.dimensions}")
    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
    thumbnail = np.array(thumbnail)
    
    resized_masks = [cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST) for mask in masks.values()]
    
    masked_images = [cv2.bitwise_and(thumbnail, thumbnail, mask=mask) for mask in resized_masks]
    return masked_images, list(masks.values())

def create_gt_tiles(arrays, tile_size):
    num_rows = int(np.ceil(arrays[0].shape[0] / tile_size))
    num_cols = int(np.ceil(arrays[0].shape[1] / tile_size))
    tiles = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            tile = [array[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] for array in arrays]
            tiles.append((row, col, tile))
    
    return tiles