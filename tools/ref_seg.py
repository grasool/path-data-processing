import torch
import argparse
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import numpy as np
from torchvision import transforms as T
import cv2
from PIL import Image
import os
import sys
import numpy as np
import openslide
sys.path.append('../path-data-processing')
from models.unet import UNet
from openslide.deepzoom import DeepZoomGenerator
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
import xml.etree.ElementTree as ET

def main(args):
    torch.cuda.empty_cache()
    os.makedirs(f'outputs/{args.wsi_dir.split("/")[1][:-4]}',exist_ok=True)
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    inp_path = args.wsi_dir
    slide = openslide.open_slide(inp_path)
    tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
    print(f"WSI Dimensions: {slide.dimensions}")

    max_dim = len(tiles.level_tiles)
    cols, rows = tiles.level_tiles[max_dim-1]   
    tile_count = 0 
    unused = []
    restitch = np.zeros(slide.dimensions)
    id2label = {
        0: "Background",
        1: "BloodV"
    }

    label2id = {v: k for k, v in id2label.items()}

#     model = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b5-finetuned-ade-640-640",
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True
# )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "./segformer_results",
            id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=False
    )
    model.eval()

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
                # import pdb; pdb.set_trace()
                print("Processing tile number:", tile_name)
                img = Image.fromarray(temp_tile_np)

                encoding = feature_extractor(img, return_tensors="pt")
                pixel_values = encoding.pixel_values.squeeze().unsqueeze(dim=0)
                
                # image = transforms_test(img)
                # image=image.unsqueeze(0).to(device='cuda')
                with torch.no_grad():
                    preds = model(pixel_values.to(device='cpu'))
                logits = preds.logits
                # labels = preds.label_ids
                predictions = np.argmax(logits.cpu(), axis=1)
                predictions = predictions.type('torch.DoubleTensor')
                predictions*=255
                # import pdb; pdb.set_trace()
                # if tile_name == "4_10":
                #     import pdb; pdb.set_trace()
                restitch[row*256:row*256+256,col*256:col*256+256] = torch.nn.functional.interpolate(predictions.unsqueeze(0),restitch[row*256:row*256+256,col*256:col*256+256].shape,mode='bilinear').squeeze(0).squeeze(0).numpy()
                # restitch[row*256:row*256+256,col*256:col*256+256] = cv2.resize(np.array(predictions).astype(np.uint8),(256,256),interpolation = cv2.INTER_LINEAR) if restitch[row*256:row*256+256,col*256:col*256+256].shape == (256,256) else cv2.resize(np.array(predictions).astype(np.uint8),restitch[row*256:row*256+256,col*256:col*256+256].shape,interpolation = cv2.INTER_LINEAR)
                
                # torch.nn.functional.interpolate(predictions.unsqueeze(0),restitch[row*256:row*256+256,col*256:col*256+256].shape,mode='bilinear')
                #tifffile giving weird bugs so i use imageio
                # temp_tile_np -> to model
            else:
                
                print("NOT PROCESSING TILE:", tile_name)
                
                # pdb.set_trace()
                

            tile_count+=1
    # print(f"Step 3/3: Making GT patches")
    thumbnail = slide.get_thumbnail(slide.level_dimensions[-1])
    thumbnail = np.array(thumbnail)
    # Resize the mask to match the thumbnail's size
    # resized_mask = cv2.resize(mask, thumbnail.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    restitch = restitch.astype(np.uint8)
    # Apply the mask to the thumbnail image
    # import pdb; pdb.set_trace()
    new_h = int(restitch.shape[0]/args.scale)
    new_w = int(restitch.shape[1]/args.scale)
    masked_image = cv2.bitwise_and(thumbnail, thumbnail, mask=restitch)
    plt.imsave(f'outputs/{args.wsi_dir.split("/")[1][:-4]}/test_inference.jpg',cv2.resize(restitch,(new_h,new_w),interpolation = cv2.INTER_LINEAR))

    plt.imsave(f'outputs/{args.wsi_dir.split("/")[1][:-4]}/masked_wsi.jpg',cv2.resize(masked_image,(new_h,new_w),interpolation = cv2.INTER_LINEAR))
    print(f"***Processed Slide***")
    #yellow: r: 255, g: 255, b: 0
    #bgr (0,255,255)
    # import pdb; pdb.set_trace()
    # Threshold the image to obtain a binary mask
    _, binary_image = cv2.threshold(restitch, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # np.stack([np.zeros((2800,2800)),restitch,restitch])
    # xml_file_path = "binary_mask.xml"
    # create_xml_from_binary_mask(restitch, xml_file_path)

    # Create the new XML structure
    new_xml_tree = create_xml(contours)

    # Save the new XML file
    new_xml_path = f'outputs/{args.wsi_dir.split("/")[1][:-4]}/new_annotations.xml'
    new_xml_tree.write(new_xml_path)

    # Print the path of the generated XML file
    print(f'XML file saved at: {new_xml_path}')
    # import pdb; pdb.set_trace()
# Fu

    # restitch = np.zeros((4608,3456))
    # num_rows = restitch.shape[0] // 224
    # num_cols = restitch.shape[1] // 224
    # for idx, img_path in enumerate(sorted_array):
    #     row_idx = idx // num_cols
    #     col_idx = idx % num_cols
        
    #     start_row = row_idx * 224
    #     start_col = col_idx * 224
    #     # import pdb; pdb.set_trace()


    #     restitch[start_row:start_row+224, start_col:start_col+224]= cv2.imread(masks_path+'/'+sorted_array[idx],0) 
    # cv2.imwrite('67.jpg',restitch)
    # import pdb; pdb.set_trace()
def create_xml(contours):
    # Create the root element with MicronsPerPixel attribute
    root = ET.Element("Annotations", MicronsPerPixel="0.494200")
    
    # Iterate through each contour and create corresponding XML elements
    for i, contour in enumerate(contours):
        annotation_elem = ET.SubElement(root, "Annotation", Id=str(i+1), Name="", ReadOnly="0", NameReadOnly="0", 
                                        LineColorReadOnly="0", Incremental="0", Type="4", LineColor="65280", 
                                        Visible="1", Selected="1", MarkupImagePath="", MacroName="")
        
        ET.SubElement(annotation_elem, "Attributes")
        regions_elem = ET.SubElement(annotation_elem, "Regions")
        ET.SubElement(regions_elem, "RegionAttributeHeaders")
        
        # Assuming length and area calculations for the Region element
        length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        region_elem = ET.SubElement(regions_elem, "Region", Id=str(i+1), Type="0", Zoom="0.450000", Selected="0", 
                                    ImageLocation="", ImageFocus="-1", Length=str(length), Area=str(area), 
                                    LengthMicrons=str(length * 0.494200), AreaMicrons=str(area * (0.494200**2)), 
                                    Text="", NegativeROA="0", InputRegionId="0", Analyze="1", DisplayId=str(i+1))
        
        ET.SubElement(region_elem, "Attributes")
        vertices_elem = ET.SubElement(region_elem, "Vertices")
        
        for point in contour:
            ET.SubElement(vertices_elem, "Vertex", X=str(point[0][0]), Y=str(point[0][1]), Z="0")
    
    return ET.ElementTree(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--wsi_dir', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--scale', type=int)
    # parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)