import argparse
import pdb
import sys
import os
import torch
sys.path.append('../path-data-processing')
from models.unet import UNet
from torch import optim, nn
from torch.utils.data import DataLoader
from data_patches.dataset import data_moffitt
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
def main(args):

    transforms_train = transforms.Compose([
        transforms.Resize(240,interpolation=InterpolationMode.BILINEAR, max_size=None),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
        transforms.RandomHorizontalFlip(),
    
        transforms.ToTensor()
    ])
    model = UNet(in_channels=3, num_classes=1)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    # import pdb; pdb.set_trace()
    svs_dir = args.data_dir + '/train_patches'
    anno_dir = args.data_dir + '/train_gt'
    dataset = data_moffitt(svs_dir,anno_dir,transforms=transforms_train)
    data_loader_train = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    num_epochs = 50
    for epoch in range(num_epochs):
        iteration = 0 
        print(f"Epoch num: {epoch}")
        for patches,masks in data_loader_train:
            # patches,masks = patches.to(device='cuda'),masks.to(device='cuda')
            optimizer.zero_grad()
            outs = model(patches) 
            loss = criterion(outs,masks)
            loss.backward()
            optimizer.step()
            print(f"Ep: {epoch}, {iteration+1}/{len(data_loader_train)} Iteration Loss: {loss.item()}")
            iteration+=1
        if(((epoch+1)%10)==0):
            torch.save(model.state_dict(),f'Cmoffit{epoch}.pth')
    print("Training Completed")
    # model.eval()
    # with torch.no_grad():
    #   for images in os.listdir
    #   import pdb; pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-c', type=str, )
    # parser.add_argument('--epochs', '-r', type=int, default=15)
    # parser.add_argument('--lr', '-r', type=float, default=3e-4)
    # parser.add_argument('--batch_size', '-r', type=int, default=1)

    parser.add_argument('--data_dir', '-c', type=str)
    # parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)