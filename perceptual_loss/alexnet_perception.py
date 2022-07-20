import os
import torch
import torch.nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torchvision.utils as utils
import cv2 
import numpy as np 
from PIL import Image

def get_multi_level_feature_map(input_img, model):
    input_img = input_img.unsqueeze(0)
    conv_results = []
    x = input_img
    for idx, operation in enumerate(model.features):
        x = operation(x)
        if idx in {1, 4, 7, 9, 11}:
            x = F.normalize(x, p=2.0, dim=1)
            conv_results.append(x)
    
    return conv_results

def mask_observed_rgb(rgb, mask):
    rgb[:,:,0] = rgb[:,:,0] * (mask / 1.0)
    rgb[:,:,1] = rgb[:,:,1] * (mask / 1.0)
    rgb[:,:,2] = rgb[:,:,2] * (mask / 1.0)
    return rgb

def measure_perceptual_similarity(pseudo_rgb, observed_rgb, observed_mask):
    masked_observed_rgb = mask_observed_rgb(observed_rgb, observed_mask)

    # step1: pre-process two input images
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),             # resize the input to 224x224
        transforms.ToTensor(),              # put the input to tensor format
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the input
        # the normalization is based on images from ImageNet
    ])

    pseudo_rgb = pseudo_rgb / (np.max(pseudo_rgb) - np.min(pseudo_rgb)) * 255
    pseudo_rgb = pseudo_rgb.astype(np.uint8)

    transformed_pseudo_rgb = data_transforms(Image.fromarray((pseudo_rgb).astype(np.uint8)))
    transformed_observed_rgb = data_transforms(Image.fromarray(masked_observed_rgb))
    
    # step2: load pre-trained alexnet model
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()

    # step3: extract multi-level feature maps
    pseudo_features = get_multi_level_feature_map(transformed_pseudo_rgb, alexnet)
    observed_features = get_multi_level_feature_map(transformed_observed_rgb, alexnet)

    # step4: measure the perceptual similarity
    perceptual_error = 0
    for i in range(len(pseudo_features)):
        pseudo = pseudo_features[i]
        observed = observed_features[i]

        spatial_norm = torch.norm((pseudo - observed).squeeze(0), dim=0)
        average_norm = torch.mean(spatial_norm).item()
        perceptual_error += average_norm
    
    return perceptual_error

if __name__ == '__main__':
    pseudo_rgb = cv2.imread('./t1.png')
    observed_rgb = cv2.imread('./t2.png')
    observed_mask = cv2.imread('./t3.png', -1)

    perceptual_error = measure_perceptual_similarity(pseudo_rgb, observed_rgb, observed_mask)
    print('error:', perceptual_error)
