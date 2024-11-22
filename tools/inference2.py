import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import cv2
import os
import pandas as pd

def dice_score(pred, target, epsilon=1e-6):
    """
    Calculate Dice score between prediction and target
    """
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return dice.item()

class TestDataset(Dataset):
    def __init__(self, images_dir, anno_dir, processor):
        self.images_dir = images_dir
        self.anno_dir = anno_dir
        self.processor = processor

        images_list = sorted(os.listdir(self.images_dir))
        annotations_list = sorted(os.listdir(self.anno_dir))
        self.data_info = pd.DataFrame({'images': images_list, 'annotations': annotations_list})

    def __getitem__(self, index):
        patch_name = self.data_info.iloc[index, 0]
        gt_name = self.data_info.iloc[index, 1]
        
        # Load image and ground truth
        patch = Image.open(os.path.join(self.images_dir, patch_name))
        gt = Image.open(os.path.join(self.anno_dir, gt_name))
        
        # Convert ground truth to numpy array
        gt_array = np.array(gt)
        
        # Create mask based on colors in your ground truth
        mask = np.zeros(gt_array.shape[:2], dtype=np.uint8)
        
        # Green regions are cancer (class 1) and red regions are atypical (class 2)
        green_mask = (gt_array[:, :, 1] > 200) & (gt_array[:, :, 0] < 100) & (gt_array[:, :, 2] < 100)
        red_mask = (gt_array[:, :, 0] > 200) & (gt_array[:, :, 1] < 100) & (gt_array[:, :, 2] < 100)
        
        mask[green_mask] = 1  # Cancer (Green)
        mask[red_mask] = 2    # Atypical (Red)
        
        # Resize mask to match model input size
        mask = cv2.resize(mask, (160, 160), cv2.INTER_NEAREST)
        
        # Process image using processor
        inputs = self.processor(images=patch, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze()
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return {
            "pixel_values": pixel_values, 
            "labels": mask,
            "image_name": patch_name
        }

    def __len__(self):
        return len(self.data_info)

def test_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model configuration
    id2label = {0: "Background", 1: "Cancer", 2: "Atypical"}
    label2id = {v: k for k, v in id2label.items()}
    
    # Create base model first
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     r"/home/ubuntu/path-data-processing/tools/segformer_results7/checkpoint-4400",
    #     id2label=id2label,
    # label2id=label2id,
    # ignore_mismatched_sizes=False
    # )
    # model.to(device)
    # model.eval()
    
    
    # Load the saved state dict
    state_dict = torch.load("/home/ubuntu/path-data-processing/tools/best_model.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Initialize image processor
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

    # Create test dataset and dataloader
    test_dataset = TestDataset(
        images_dir=r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_patches',
        anno_dir=r'/home/ubuntu/path-data-processing/data_patches/testing_dataset(v3)/train_gt',
        processor=processor
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Found {len(test_dataset)} test images")

    # Initialize metrics
    dice_scores = {
        'background': [],
        'cancer': [],
        'atypical': []
    }

    # Create directory for saving predictions
    os.makedirs('prediction_results', exist_ok=True)

    # Testing loop
    print("Starting inference...")
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx % 10 == 0:
                print(f"Processing image {idx}/{len(test_loader)}")
                
            # Move inputs to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            image_names = batch["image_name"]

            # Get model predictions
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Convert logits to predictions
            predictions = torch.argmax(logits, dim=1)

            # Calculate Dice score for each class
            for class_idx in range(3):
                class_pred = (predictions == class_idx).float()
                class_target = (labels == class_idx).float()
                dice = dice_score(class_pred, class_target)
                
                if class_idx == 0:
                    dice_scores['background'].append(dice)
                elif class_idx == 1:
                    dice_scores['cancer'].append(dice)
                else:
                    dice_scores['atypical'].append(dice)

            # Save predictions as images
            pred_np = predictions[0].cpu().numpy()
            save_path = os.path.join('prediction_results', image_names[0])
            
            # Convert predictions to RGB image for visualization
            pred_rgb = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
            pred_rgb[pred_np == 1] = [0, 255, 0]  # Cancer in green
            pred_rgb[pred_np == 2] = [255, 0, 0]  # Atypical in red
            
            cv2.imwrite(save_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    # Calculate and print average Dice scores
    avg_dice_scores = {
        'background': np.mean(dice_scores['background']),
        'cancer': np.mean(dice_scores['cancer']),
        'atypical': np.mean(dice_scores['atypical'])
    }
    
    print("\nAverage Dice Scores:")
    for class_name, score in avg_dice_scores.items():
        print(f"{class_name}: {score:.4f}")
    
    # Save metrics to file
    with open('prediction_results/metrics.txt', 'w') as f:
        f.write("Average Dice Scores:\n")
        for class_name, score in avg_dice_scores.items():
            f.write(f"{class_name}: {score:.4f}\n")
    
    return avg_dice_scores

if __name__ == "__main__":
    test_model()