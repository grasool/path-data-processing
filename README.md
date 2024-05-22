﻿# Process Pathology Slides
 ### Create patches from given WSI
 ### Create annotation patches
 ### Train model
 ### Run inference on the trained model
 ### Visualize results

## Setting Up Environment for Windows Machines

1. Create conda environment
```conda create --name path-process python=3.10```
2. Activate conda environment
```conda activate path-process```
3. Install pytorch
```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
4. Install openslide
```pip install openslide-python```
5. Download and copy openslide binaries from [here](https://openslide.org/api/python/#basic-usage)
6. Update path-to-baniries and test openslide using ```test-open-slide.py```
7. Install other requirements
```pip install -r requirements.txt```
