import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, TrainingArguments
from torchvision.transforms import transforms as T
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import wandb
from typing import Dict, List
import evaluate
from sklearn.metrics import f1_score

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ImprovedDataMoffittSeg(Dataset):
    def __init__(self, images_dir: str, anno_dir: str, feature_extractor, transforms=None, phase='train'):
        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.feature_extractor = feature_extractor
        self.phase = phase
        
        # Define image size constants
        self.input_size = (160, 160)  # Input image size
        self.target_size = (40, 40)   # Target mask size (1/4 of input size due to model architecture)
        
        # Separate transforms for images and masks
        if transforms and phase == 'train':
            self.image_transforms = transforms
            self.mask_transforms = T.Compose([
                T.RandomResizedCrop(size=self.input_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=45),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ])
        else:
            self.image_transforms = T.Compose([
                T.Resize(self.input_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transforms = T.Compose([
                T.Resize(self.input_size)
            ])

        images_list = sorted(os.listdir(self.images_dir))
        annotations_list = sorted(os.listdir(self.anno_dir))
        self.data_info = pd.DataFrame({'images': images_list, 'annotations': annotations_list})

    def __getitem__(self, index):
        patch_name = self.data_info.iloc[index, 0]
        gt_name = self.data_info.iloc[index, 1]
        
        # Load image and ground truth
        patch = Image.open(os.path.join(self.images_dir, patch_name))
        gt = Image.open(os.path.join(self.anno_dir, gt_name))
        
        # Convert to numpy arrays
        patch_array = np.array(patch)
        gt_array = np.array(gt)
        
        # Create mask
        mask = np.zeros(gt_array.shape[:2], dtype=np.uint8)
        green_mask = (gt_array[:, :, 1] > 200) & (gt_array[:, :, 0] < 100) & (gt_array[:, :, 2] < 100)
        red_mask = (gt_array[:, :, 0] > 200) & (gt_array[:, :, 1] < 100) & (gt_array[:, :, 2] < 100)
        mask[green_mask] = 1  # Cancer
        mask[red_mask] = 2    # Atypical
        
        # Convert to PIL for transforms
        patch_pil = Image.fromarray(patch_array)
        mask_pil = Image.fromarray(mask)
        
        if self.phase == 'train':
            # Get random parameters for consistent transforms
            seed = torch.randint(0, 2**32, (1,))[0].item()
            torch.manual_seed(seed)
            patch_transformed = self.image_transforms(patch_pil)
            
            torch.manual_seed(seed)
            mask_pil = self.mask_transforms(mask_pil)
        else:
            patch_transformed = self.image_transforms(patch_pil)
            mask_pil = self.mask_transforms(mask_pil)
        
        # Convert mask to tensor and resize to target size
        mask = torch.from_numpy(np.array(mask_pil))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                           size=self.target_size, 
                           mode='nearest').squeeze().long()

        return {
            "pixel_values": patch_transformed,
            "labels": mask,
            "image_name": patch_name
        }

    def __len__(self):
        return len(self.data_info)

# class ImprovedDataMoffittSeg(Dataset):
#     def __init__(self, images_dir: str, anno_dir: str, feature_extractor, transforms=None, phase='train'):
#         self.images_dir = images_dir
#         self.anno_dir = anno_dir
#         self.transforms = transforms
#         self.feature_extractor = feature_extractor
#         self.phase = phase

#         images_list = sorted(os.listdir(self.images_dir))
#         annotations_list = sorted(os.listdir(self.anno_dir))
#         self.data_info = pd.DataFrame({'images': images_list, 'annotations': annotations_list})

#     def __getitem__(self, index):
#         patch_name = self.data_info.iloc[index, 0]
#         gt_name = self.data_info.iloc[index, 1]
        
#         # Load image and ground truth
#         patch = Image.open(os.path.join(self.images_dir, patch_name))
#         gt = Image.open(os.path.join(self.anno_dir, gt_name))
        
#         # Convert to numpy arrays
#         patch_array = np.array(patch)
#         gt_array = np.array(gt)
        
#         # Create mask
#         mask = np.zeros(gt_array.shape[:2], dtype=np.uint8)
#         green_mask = (gt_array[:, :, 1] > 200) & (gt_array[:, :, 0] < 100) & (gt_array[:, :, 2] < 100)
#         red_mask = (gt_array[:, :, 0] > 200) & (gt_array[:, :, 1] < 100) & (gt_array[:, :, 2] < 100)
#         mask[green_mask] = 1  # Cancer
#         mask[red_mask] = 2    # Atypical
        
#         # Apply augmentations
#         if self.transforms and self.phase == 'train':
#             # Convert to PIL for transforms
#             patch_pil = Image.fromarray(patch_array)
#             mask_pil = Image.fromarray(mask)
            
#             # Get random parameters for consistent transforms
#             seed = torch.randint(0, 2**32, (1,))[0].item()
#             torch.manual_seed(seed)
#             patch_transformed = self.transforms(patch_pil)
            
#             torch.manual_seed(seed)
#             mask = self.transforms(mask_pil)
            
#         else:
#             # Validation/Test phase
#             patch_transformed = T.Compose([
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])(patch)
#             mask = torch.from_numpy(mask)

#         # Resize mask to match model requirements
#         mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
#                            size=(160, 160), 
#                            mode='nearest').squeeze().long()

#         return {
#             "pixel_values": patch_transformed,
#             "labels": mask,
#             "image_name": patch_name
#         }

#     def __len__(self):
#         return len(self.data_info)

class ImprovedTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int = 100,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.patience = patience
        
        # Initialize tracking variables
        self.best_cancer_dice = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
        # Calculate class weights based on pixel distribution
        self.class_weights = self._calculate_pixel_weights()
        
        # Initialize loss function
        self.criterion = FocalLoss(gamma=2, alpha=self.class_weights.to(device))
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize metrics
        self.metric = evaluate.load("mean_iou")

    def _calculate_pixel_weights(self) -> torch.Tensor:
        """Calculate class weights based on pixel distribution in training set."""
        class_pixels = torch.zeros(3)
        for batch in self.train_loader:
            labels = batch["labels"]
            for i in range(3):
                class_pixels[i] += (labels == i).sum().item()
        
        total_pixels = class_pixels.sum()
        weights = total_pixels / (3 * class_pixels)
        return weights / weights.sum()  # Normalize weights

    def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Create a schedule with a learning rate that decreases following the values of the cosine function."""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _dice_score(self, pred: torch.Tensor, target: torch.Tensor, class_idx: int) -> float:
        """Calculate Dice score for a specific class."""
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        return (2. * intersection + 1e-6) / (pred_class.sum() + target_class.sum() + 1e-6)

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            # Move data to device
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values)
            loss = self.criterion(outputs.logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to wandb
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": self.scheduler.get_last_lr()[0],
                "global_step": self.global_step
            })
        
        return total_loss / len(self.train_loader)

    def validate(self) -> Dict[str, float]:
        """Validate the model and return metrics."""
        self.model.eval()
        dice_scores = {
            "background": [], "cancer": [], "atypical": []
        }
        val_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values)
                val_loss += self.criterion(outputs.logits, labels).item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                
                # Calculate dice scores for each class
                for class_idx, class_name in enumerate(["background", "cancer", "atypical"]):
                    dice = self._dice_score(predictions, labels, class_idx)
                    dice_scores[class_name].append(dice)
        
        # Calculate average scores
        avg_scores = {
            k: sum(v) / len(v) for k, v in dice_scores.items()
        }
        avg_scores["val_loss"] = val_loss / len(self.val_loader)
        
        return avg_scores

    def train(self):
        """Main training loop."""
        wandb.init(project="improved-segformer-segmentation")
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            metrics = self.validate()
            
            # Logging
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics
            })
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            
            # Early stopping based on cancer dice score
            if metrics["cancer"] > self.best_cancer_dice:
                self.best_cancer_dice = metrics["cancer"]
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model_cancer.pth")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

# Training transforms with additional augmentations
train_transforms = T.Compose([
    T.RandomResizedCrop(size=(160, 160), scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=45),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
IMAGE_PATH = r'/home/ubuntu/path-data-processing/data_patches/new_training/train_patches'
MASK_PATH = r'/home/ubuntu/path-data-processing/data_patches/new_training/train_gt'
VAL_IMAGE_PATH = r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_patches'
VAL_MASK_PATH = r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_gt'

# Usage example:
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    
    # Create datasets
    train_dataset = ImprovedDataMoffittSeg(
        images_dir=IMAGE_PATH,
        anno_dir=MASK_PATH,
        feature_extractor=feature_extractor,
        transforms=train_transforms,
        phase='train'
    )
    
    val_dataset = ImprovedDataMoffittSeg(
        images_dir=VAL_IMAGE_PATH,
        anno_dir=VAL_MASK_PATH,
        feature_extractor=feature_extractor,
        transforms=None,
        phase='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=3,
        id2label={0: "Background", 1: "Cancer", 2: "Atypical"},
        label2id={"Background": 0, "Cancer": 1, "Atypical": 2},
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Initialize trainer
    trainer = ImprovedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100,
        patience=10
    )
    
    # Start training
    trainer.train()