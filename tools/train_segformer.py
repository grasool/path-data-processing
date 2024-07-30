import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from sklearn.metrics import f1_score
import numpy as np
from torchvision import transforms as T
import cv2
from PIL import Image,ImageOps
import os
import pandas as pd

torch.cuda.empty_cache()

class data_moffitt_seg(Dataset):

    def __init__(self,images_dir,anno_dir,feature_extractor,transforms=None):

        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.feature_extractor = feature_extractor

        images_list = os.listdir(self.images_dir)
        images_list = sorted(images_list)
        annotations_list =  os.listdir(self.anno_dir)
        annotations_list = sorted(annotations_list)
        
        self.data_info  = pd.DataFrame({'images':images_list,'annotations':annotations_list})


    def __getitem__(self,index):
        # pdb.set_trace()
        patch_name = self.data_info.iloc[index,0]
        gt_name = self.data_info.iloc[index,1]
        patch = Image.open(os.path.join(self.images_dir,patch_name))
        gt = Image.open(os.path.join(self.anno_dir,gt_name))

        gt= ImageOps.grayscale(gt) 
        
        mask = cv2.resize(np.array(gt),(160,160),cv2.INTER_CUBIC)
        mask = np.array(mask)
        indices = np.where(mask>0)
        mask[indices] = 255
        mask = Image.fromarray(np.array(mask))
        # else:
        # mask = Image.open(os.path.join(self.mask_path, mask_name)).convert('L')
        encoding = self.feature_extractor(patch, return_tensors="pt")
        pixel_values = encoding.pixel_values.squeeze()
        mask = torch.from_numpy(np.array(mask)).long()/255
        mask=mask.long()
        # import pdb; pdb.set_trace()
        sample =  {"pixel_values": pixel_values, "labels": mask}
        # import pdb; pdb.set_trace()
        return sample
    def __len__(self):
        
        return len(self.data_info)
    
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b5")


# accelerator = Accelerator()

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

IMAGE_PATH = '/home/afridi/Desktop/moffitt_ali/data_patches/train_patches'
MASK_PATH = '/home/afridi/Desktop/moffitt_ali/data_patches/train_gt'

train_transform = T.Compose([
    # T.Resize(512, 512),
    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean, std)
])

val_transform = T.Compose([
    # T.Resize(512, 512),
    T.ToTensor(),
    T.Normalize(mean, std)
])

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# device = accelerator.device

#datasets
train_dataset = data_moffitt_seg(IMAGE_PATH, MASK_PATH, feature_extractor,train_transform)
# val_dataset = RustDataset('/home/Hirra/SegFormer/data/NWRD/images/validation', '/home/Hirra/SegFormer/data/NWRD/annotations/validation', mean, std, feature_extractor,train_transform)
print(f"Length of train dataset: {len(train_dataset)}")
# print(f"Length of val dataset: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Check dataloader lengths
print(f"Number of batches in train loader: {len(train_loader)}")
# print(f"Number of batches in val loader: {len(val_loader)}")
# import pdb; pdb.set_trace()
# val_set = RustDataset(IMAGE_PATH, MASK_PATH, mean, std, val_transform)

# train_dataset = CustomDataset(train_images, train_masks, feature_extractor)
# val_dataset = CustomDataset(val_images, val_masks, feature_extractor)

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Load the model
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
id2label = {
    0: "Background",
    1: "BloodV"
}

label2id = {v: k for k, v in id2label.items()}

# import json
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

# Define metrics
metric = load_metric("mean_iou")
def convert_np_to_list(data):
    if isinstance(data, dict):
        return {key: convert_np_to_list(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(logits, axis=1)
    # import pdb; pdb.set_trace()

    iou = metric.compute(predictions=predictions, references=labels, num_labels=model.config.num_labels, ignore_index=255)
    f1 = f1_score(labels.flatten(), predictions.flatten(), average='weighted')
    iou["f1"] = f1
    converted_dict = convert_np_to_list(iou)
    # import pdb; pdb.set_trace()
    # iou["eval_per_category_iou"] = iou["eval_per_category_iou"].tolist() 
    # iou["eval_per_category_accuracy"] = iou["eval_per_category_accuracy"].tolist()
    # plt.save(predictions)
    print(f"f1 score: {f1}")
    return converted_dict
# import pdb; pdb.set_trace()
# Define training arguments
training_args = TrainingArguments(
    output_dir="./segformer_results",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=1,
    num_train_epochs=10,
    weight_decay=0.01,
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    # device = "cpu",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    # device = 0,
    # compute_metrics=compute_metrics,
)

# Train and evaluate
# import pdb; pdb.set_trace()

trainer.train()
import pdb; pdb.set_trace()

# eval_results = trainer.evaluate()
# print(eval_results)
