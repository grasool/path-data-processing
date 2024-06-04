# Process Pathology Slides
 ### 1. Create patches from given WSI
 ### 2. Create annotation patches
 ### 3. Train, transfer learn or fine-tune models
 ### 4. Run inference on the trained model
 ### 5. Visualize results

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


# To-do @Ali and @Ashwin
1. Code to re-stitch and visualize the pathes and their annotaitons in Aperio Imagescopr
2. Code to identify patches that are tissue but have no groud truth.
3. Model training and validation using the created dataset
4. Model testing and inference
5. Result visualiztion
 
