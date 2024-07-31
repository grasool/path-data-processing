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
      <img src="tools/aperio3.png" height="270">
    </td>
  </tr>
</table>

# 🚀 Updates

- [2024.07.31] Added support for Segformer-B5 based Training and Inference!
- [2024.07.31] Results Visualization in Aperio Image Scope 🔥
- [2024.06.10] Added support for SAM based Training and Inference!
- [2024.06.03] Added support for Unet based Training and Inference!

# 📍 Implementations

 ##### 1. Create annotation and image patches from given WSI 
 ##### 2. Train, transfer learn or fine-tune models 
 ##### 3. Run inference on the trained model 
 ##### 4. Restitch Inference patches and Visualize results 

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

 ## 1. Create image and annotation patches from given WSI 
 
 <p align="center"> <img src="tools/image2.png" height="160"\></p>
 <p align="center"> <img src="tools/gt2.png" height="150"\></p>
 
Put ```.tif``` or ```.svs``` slides for training and/or testing purposes in ```data_wsi/images/train```, ```data_wsi/images/test``` folders

Put xml annotations in ```data_wsi/annotations/train```, ```data_wsi/annotations/test``` folders. 

Make sure the WSI name in ```data_wsi/annotations/train``` or ```data_wsi/annotations/test```,  matches its xml file name. For example: ```TMA457_1_1_C_.tif``` file is present in ```data_wsi/images/train```, it's corresponding ```TMA457_1_1_C_.xml``` must be present in ```data_wsi/annotations/train```

White backgound based patches will be filtered and dumped in the folder ```data_patches/np```
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


 ## 2. Train, transfer learn or fine-tune models 
##### Segformer B5 Training

Additional library is required to train SegformerB5. Run ```pip install transformers[torch]```

Change image/annotations path in line 70,71 in ```tools/train_segformer.py``` 

Batch size can be changed from line 158 in ```tools/train_segformer.py``` [Optional]

Start the training by runnning:
```bash
python tools/train_segformer.py
```
Weights of Segformer will be saved in ```./segformer_results``` folder
##### Unet Training
To start training UNet on ```data_patches/train_patches```, ```data_patches/train_gt```, run:
```bash
python tools/train.py --data_dir ./data_patches
```
Weights will be saved in root directory.
## 3. Run inference on the trained model 
<p align="center">
  <img src="tools/imagem.png" height="300">
  <img src="tools/test_inference.jpg" height="300">
</p>

Put all WSIs to test in ```tools``` folder.

##### Segformer B5 Inference
To use weights of segformer, Run:
```bash
python tools/ref_seg.py --wsi_dir tools/wsi.tif --checkpoint ./path-data-processing/segformer_results --scale 1
```
This will save restitched model's output prediction patches and xml annotations for visualization in ```output/``` folder
--scale "n" argument will divide the restitched image by "n" for 1x, 2x, nx scale.
##### Unet based Inference

To use weights of UNet, Run:
```bash
python tools/inference_wsi_unet.py --wsi_dir tools/wsi.tif --checkpoint ./path/to/unet.pth --scale 1
```
##### SAM based Inference 
To be updated by @Ashwin

### 4. Aperio Image Scope Visualization
After running the inferences based on either Segformer or Unet, xml files will be stored in ```outputs/``` folder. In APerio, load your tiff image and navigate to View -> Annotations -> Open Local Annotations -> Select our generated xml file.

<p align="center">
  <img src="tools/aperio.png" height="260">
  <img src="tools/aperio2.png" height="260">
</p>

# Contributing 🤝
Pull Requests are welcome ❤️. Feel free to contribute if you have amazing ideas! 
