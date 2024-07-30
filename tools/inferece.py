import sys
import os
sys.path.append('../path-data-processing')
from models.unet import UNet
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imsave('test_inference.jpg',reverse_transforms(image[0]))

def main(args):
    transforms_test = transforms.Compose([
        transforms.Resize(240,interpolation=InterpolationMode.BILINEAR, max_size=None),
        transforms.ToTensor()
    ])

    images_path = '/home/afridi/Desktop/moffitt_ali/data_patches/train_patches'
    model = UNet(in_channels=3, num_classes=1).to(device='cpu')
    checkpoint = torch.load(args.checkpoint)
    # import pdb; pdb.set_trace()
    model.load_state_dict(checkpoint)
    # model= model.to(device='cuda')
    model.eval()

    for images in os.listdir(images_path):
        image = Image.open(images_path+'/'+images)
        # image_np = np.array(images)
        # image_np = image_np.transpose(2,0,1)
        # image_np=torch.tensor(image_np).squeeze(0)
        image = transforms_test(image)
        # image=image.unsqueeze(0).to(device='cuda')

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            outs = model(image.unsqueeze(0))
        show_tensor_image(outs)    
        print(f"Processed tile {images}")
        import pdb; pdb.set_trace()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-c', type=str, )
    # parser.add_argument('--epochs', '-r', type=int, default=15)
    # parser.add_argument('--lr', '-r', type=float, default=3e-4)
    # parser.add_argument('--batch_size', '-r', type=int, default=1)

    parser.add_argument('--checkpoint', '-c', type=str)
    # parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)