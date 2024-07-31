# HistoPathology Slides Processing Pipeline
This repository is a complete package which deals with Whole Slide Histopathological Images. We perform custom user demanded segmentations using state of the art machine learning models on WSI! We also visualize results on eminent user friendly softwares for pathologists (Aperio)
<table>
  <tr>
    <td style="text-align: center;">
      <p><strong>Input Slide</strong></p>
      <img src="tools/image1.png" height="300">
    </td>
    <td style="text-align: center;">
      <p><strong>Restitched Inference</strong></p>
      <img src="tools/test_inference2.jpg" height="270">
    </td>
    <td style="text-align: center;">
      <p><strong>Visualization in Aperio</strong></p>
      <img src="tools/aperio3.png" height="250">
    </td>
  </tr>
</table>

# 📍 Implementations

 ### 1. Create annotation and image patches from given WSI 
 ### 2. Train, transfer learn or fine-tune models 
 ### 3. Run inference on the trained model 
 ### 4. Visualize results 

## Setting Up Environment for Windows Machines
1. Create conda environment
   
```bash
conda create --name path-process python=3.10
```
2. Activate conda environment
```bash
conda activate path-process
```
3. Install pytorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
4. Install openslide
```bash
pip install openslide-python
```
5. Download and copy openslide binaries from [here](https://openslide.org/api/python/#basic-usage)
6. Update path-to-baniries and test openslide using
```bash
test-open-slide.py
```
7. Install other requirements
```bash
pip install -r requirements.txt
```
# 📍 Example Usage

 ### 1. Create image and annotation patches from given WSI 
 
 <p align="center"> <img src="tools/image2.png" height="100"\></p>
 <p align="center"> <img src="tools/gt2.png" height="100"\></p>
 
Put .tif or .svs slides for training and/or testing purposes in data_wsi/images/train, data_wsi/images/test folders

Put xml annotations in data_wsi/annotations/train, data_wsi/annotations/test 

Make sure the WSI name in ```data_wsi/annotations/train``` or ```data_wsi/annotations/test```,  matches its xml file name. For example: ```TMA457_1_1_C_.tif``` file is present in ```data_wsi/images/train```, it's corresponding ```TMA457_1_1_C_.xml``` must be present in ```data_wsi/annotations/train```

##### Train image and annotations patches

To generate image patches by ignoring white background for training, use:

 ```bash
python data_wsi/preprocess_train.py --wsi_path data_wsi/images/train --xml_path data_wsi/annotations/train --target_dir ./data_patches
```
##### Test image and annotations patches

To generate image patches by ignoring white background for test, use:

 ```bash
python data_wsi/preprocess_test.py --wsi_path data_wsi/images/test --xml_path data_wsi/annotations/test --target_dir ./data_patches
```
<p align="center">
  <img src="tools/imagem.png" height="300">
  <img src="tools/test_inference.jpg" height="300">
</p>


<p align="center">
  <img src="tools/aperio.png" height="300">
  <img src="tools/aperio2.png" height="300">
</p>
 
