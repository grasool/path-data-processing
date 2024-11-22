import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments, Trainer
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from datasets import load_dataset
from sklearn.metrics import f1_score
import numpy as np
from torchvision import transforms as T
import cv2
from PIL import Image
import os
import pandas as pd
import evaluate
import wandb
from torch.nn import CrossEntropyLoss
from transformers.trainer_utils import EvalPrediction
import matplotlib.pyplot as plt



# Clear GPU cache
torch.cuda.empty_cache()

# Augmentation function for underrepresented red-class regions
def augment_typical_class(image, mask):
    if (mask == 2).sum() > 500:  # Augment if enough red-class pixels
        augmented_image = T.RandomHorizontalFlip(p=0.5)(image)
        augmented_image = T.RandomRotation(degrees=30)(augmented_image)
        return augmented_image, mask
    return image, mask

# Dataset definition
class DataMoffittSeg(Dataset):
    def __init__(self, images_dir, anno_dir, feature_extractor, transforms=None):
        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.feature_extractor = feature_extractor

        images_list = sorted(os.listdir(self.images_dir))
        annotations_list = sorted(os.listdir(self.anno_dir))
        self.data_info = pd.DataFrame({'images': images_list, 'annotations': annotations_list})

    def __getitem__(self, index):
        patch_name = self.data_info.iloc[index, 0]
        gt_name = self.data_info.iloc[index, 1]
        
        patch = Image.open(os.path.join(self.images_dir, patch_name))
        gt = Image.open(os.path.join(self.anno_dir, gt_name))
        gt_array = np.array(gt)
        
        mask = np.zeros(gt_array.shape[:2], dtype=np.uint8)
        green_mask = (gt_array[:, :, 0] < 100) & (gt_array[:, :, 1] > 200) & (gt_array[:, :, 2] < 100)
        red_mask = (gt_array[:, :, 0] > 200) & (gt_array[:, :, 1] < 100) & (gt_array[:, :, 2] < 100)
        
        mask[green_mask] = 1  # Cancer
        mask[red_mask] = 2    # Atypical
        
        # Augmentation logic for red-class
        if self.transforms:
            patch, mask = augment_typical_class(patch, mask)

        # Resize mask and convert image
        mask = cv2.resize(mask, (160, 160), cv2.INTER_NEAREST)
        encoding = self.feature_extractor(patch, return_tensors="pt")
        pixel_values = encoding.pixel_values.squeeze()
        mask = torch.from_numpy(mask).long()
        
        return {"pixel_values": pixel_values, "labels": mask}

    def __len__(self):
        return len(self.data_info)

# Feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

# Paths
IMAGE_PATH = r'/home/ubuntu/path-data-processing/data_patches/new_training/train_patches'
MASK_PATH = r'/home/ubuntu/path-data-processing/data_patches/new_training/train_gt'
VAL_IMAGE_PATH = r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_patches'
VAL_MASK_PATH = r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_gt'

# Define transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomResizedCrop(size=(160, 160), scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=30),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean, std)
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Dataset and DataLoader
train_dataset = DataMoffittSeg(IMAGE_PATH, MASK_PATH, feature_extractor, train_transform)
val_dataset = DataMoffittSeg(VAL_IMAGE_PATH, VAL_MASK_PATH, feature_extractor, val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model and labels
id2label = {0: "Background", 1: "Cancer", 2: "Atypical"}
label2id = {v: k for k, v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

# Define metrics
metric = evaluate.load("mean_iou")

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(logits, axis=1)

    iou = metric.compute(predictions=predictions, references=labels, num_labels=model.config.num_labels, ignore_index=255)
    f1 = f1_score(labels.flatten(), predictions.flatten(), average='weighted', labels=[0, 1, 2])
    iou["f1"] = f1
    return iou

# Weighted loss function
class_weights = torch.tensor([1.0, len(train_dataset) / 129, len(train_dataset) / 46]).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights)

# Training arguments
training_args = TrainingArguments(
    output_dir="./segformer_results7",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    num_train_epochs=100,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=3,
    logging_dir="./logs",
    report_to="wandb"
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_losses = []
        self.validation_losses = []
        self.step_count = 0
        self.top_train_losses = []  # Store top 5 lowest training losses
        self.best_val_loss = float("inf")
        self.best_model = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        # Track training loss
        self.training_losses.append(loss.item())
        self.step_count += 1
        
        # Save model with the lowest training loss
        if len(self.top_train_losses) < 5:
            self.top_train_losses.append((loss.item(), model.state_dict()))
            self.top_train_losses.sort(key=lambda x: x[0])
        elif loss.item() < self.top_train_losses[-1][0]:
            self.top_train_losses[-1] = (loss.item(), model.state_dict())
            self.top_train_losses.sort(key=lambda x: x[0])

        # Calculate validation loss every 4 steps
        if self.step_count % 4 == 0:
            self._evaluate_validation_loss()

        return (loss, outputs) if return_outputs else loss

    def _evaluate_validation_loss(self):
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.args.device)
                labels = batch["labels"].to(self.args.device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                val_loss += loss_fn(logits, labels).item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        self.validation_losses.append(avg_val_loss)
        
        # Log validation loss and save the best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_model = model.state_dict()

        # Log validation loss
        if self.args.report_to and "wandb" in self.args.report_to:
            wandb.log({"validation_loss": avg_val_loss})

        model.train()

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

        # Plot training and validation loss
        self.plot_losses()

    def plot_losses(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.training_losses, label="Training Loss", alpha=0.7)
        plt.plot(range(4, 4 * len(self.validation_losses) + 1, 4), self.validation_losses, label="Validation Loss", alpha=0.7)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.grid(True)
        plt.show()





# Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics
# )
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
print(len(train_dataset))
steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
if len(train_dataset) % training_args.per_device_train_batch_size != 0:
    steps_per_epoch += 1

total_steps = steps_per_epoch * training_args.num_train_epochs
print(f"Total Steps: {total_steps}")




# Initialize W&B and start training
wandb.init(project="segformer-semantic-segmentation")
trainer.train()

# Save best validation model
if trainer.best_model:
    torch.save(trainer.best_model, "best_model.pth")

# Save top 5 lowest training loss models
for i, (loss, model_state) in enumerate(trainer.top_train_losses):
    torch.save(model_state, f"top_train_loss_model_{i+1}.pth")