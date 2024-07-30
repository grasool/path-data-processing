import cv2
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
#
import openslide
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
import pdb
from PIL import Image
class data_moffitt(Dataset):

    def __init__(self,images_dir,anno_dir,transforms=None):

        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        
        images_list = os.listdir(self.images_dir)
        images_list = sorted(images_list)
        annotations_list =  os.listdir(self.anno_dir)
        annotations_list = sorted(annotations_list)
        
        self.data_info  = pd.DataFrame({'images':images_list,'annotations':annotations_list})


    def __getitem__(self,index):
        slide_name = self.data_info.iloc[index,0]
        anno_name = self.data_info.iloc[index,1]
        # slide = openslide.OpenSlide(os.path.join(self.images_dir,slide_name))
        # slide = Image.open(os.path.join(self.images_dir,slide_name))
        # slide = openslide.ImageSlide(slide)
        slide = openslide.open_slide(os.path.join(self.images_dir,slide_name))
        tree = ET.parse(os.path.join(self.anno_dir,anno_name))
        root = tree.getroot()
        # Define the dimensions for the mask (it should match the dimensions of the WSI)
        mask = np.zeros(slide.level_dimensions[0][::-1], dtype=np.uint8)
        # pdb.set_trace()
        # Extract annotations and draw them on the mask
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
        pdb.set_trace()
        return slide,mask
    def __len__(self):
        
        return len(self.data_info)
    