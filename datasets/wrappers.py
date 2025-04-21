import numpy as np
# from matplotlib import pyplot as plt
import re
import cv2
import torch
import models
import random
import json

import albumentations as A
# import matplotlib.pyplot as plt
import kornia as K
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from datasets import register
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
import os

@register('lpsrgan')
class SR_paired_images_wrapper_lp(Dataset):
    def __init__(
            self,
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            lbp = False,
            EIR = False,
            test = False,
            dataset=None,
            ):
        
        self.imgW = imgW
        self.imgH = imgH
        self.background = eval(background)
        self.dataset = dataset
        self.aug = aug
        self.ar = image_aspect_ratio
        self.lbp = lbp
        self.EIR = EIR
        self.test = test
        self.r_padding= False
        self.transform = A.OneOf(
            [
                A.SafeRotate(limit=2, border_mode=cv2.BORDER_REPLICATE, value=127, p=1.0),  # Rotate up to 15 degrees
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.NoOp(p=0.1)
            ],
            p=1.0,  # Ensure one of these is always selected
        )
        
        self.transform = A.Compose(
            [self.transform],  # Wrap in Compose to manage `image2`
            additional_targets={"image2": "image"},
            is_check_shapes=False,  # Disable shape check
        )
    
    def Open_image(self, img, cvt=True):
        img = cv2.imread(img)
        if cvt is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def crop_to_valid_region(self, img, mask):
        """
        Crop the image and mask to the valid region, excluding padded areas.
        
        Args:
            img (np.array): The augmented image.
            mask (np.array): The corresponding binary mask where 1 indicates valid region and 0 indicates padding.
        
        Returns:
            (np.array, np.array): Cropped image and mask.
        """
        # Find the coordinates of the valid region in the mask
        valid_pixels = np.where(mask == 1)
        
        if len(valid_pixels[0]) == 0:
            # print("Warning: No valid pixels found in the mask.")
            return img, mask  # Return the original image if no valid region is found
    
        # Get the bounding box of the valid region
        min_y, max_y = np.min(valid_pixels[0]), np.max(valid_pixels[0])
        min_x, max_x = np.min(valid_pixels[1]), np.max(valid_pixels[1])
    
        # Crop the image and mask to the bounding box
        img_cropped = img[min_y:max_y + 1, min_x:max_x + 1]
        mask_cropped = mask[min_y:max_y + 1, min_x:max_x + 1]
    
        return img_cropped, mask_cropped
    
    def pad_with_mask(self, img, min_ratio, max_ratio, color=(0, 0, 0)):
        """
        This function pads the image and creates a corresponding mask (same size as image)
        with the same padding applied to both.
        
        Parameters:
        - img: The input image.
        - min_ratio: The minimum aspect ratio to be achieved after padding.
        - max_ratio: The maximum aspect ratio to be achieved after padding.
        - color: The color to be used for padding (default is black).
        
        Returns:
        - img: The padded image.
        - mask: The generated mask with the same padding as the image.
        - border_w: The horizontal padding applied.
        - border_h: The vertical padding applied.
        """
        img_h, img_w = np.shape(img)[:2]
        
        # Initial padding values
        border_w = 0
        border_h = 0
        ar = float(img_w) / img_h
        
        # Create a mask of the same size as the image, initialized with zeros (black)
        mask = np.ones((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
        
        if ar >= min_ratio and ar <= max_ratio:
            # If aspect ratio is within desired range, return the image as is
            # Create a black mask of the same size
            mask = np.zeros_like(img, dtype=np.uint8)
            return img, mask, border_w, border_h
        
        # Adjust the aspect ratio by padding
        if ar < min_ratio:
            while ar < min_ratio:
                border_w += 1
                ar = float(img_w + border_w) / (img_h + border_h)
        else:
            while ar > max_ratio:
                border_h += 1
                ar = float(img_w) / (img_h + border_h)
        
        # Apply half padding on each side (to ensure symmetric padding)
        border_w = border_w // 2
        border_h = border_h // 2
        
        # Pad the image
        img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=color)
        # Apply the same padding to the mask as the image (mask stays black)
        mask = cv2.copyMakeBorder(mask, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=0)
        
        return img, mask, border_w, border_h
    
    def augment_images(self, lr_img, hr_img, augmentations, target_h, target_w, pad_value=127):
        """
        Apply the same augmentations to both `lr` and `hr` images.
        Args:
            lr_img (np.array): Low-resolution image.
            hr_img (np.array): High-resolution image.
            augmentations (A.Compose): Albumentations pipeline.
            target_h (int): Target height for padding.
            target_w (int): Target width for padding.
            pad_value (int): Padding value (default: 127).
        Returns:
            Tuple: Augmented `lr` and `hr` images.
        """
        # Pad both images and generate masks
        lr_padded, lr_mask, _, _ = self.pad_with_mask(lr_img, self.ar - 0.15, self.ar + 0.15, color=(127, 127, 127))
        hr_padded, hr_mask, _, _ = self.pad_with_mask(hr_img, self.ar - 0.15, self.ar + 0.15, color=(127, 127, 127))
    
        # Apply augmentations to both images and the LR mask
        augmented = augmentations(image=lr_padded, image2=hr_padded, mask=lr_mask, mask2=hr_mask)
        
        lr_aug = augmented["image"]
        hr_aug = augmented["image2"]
        lr_mask_aug = augmented["mask"]
        hr_mask_aug = augmented["mask2"]
    
        # Crop back to valid regions using the augmented mask
        lr_final, _ = self.crop_to_valid_region(lr_aug, lr_mask_aug)
        hr_final, _ = self.crop_to_valid_region(hr_aug, hr_mask_aug)  # Use the same mask for alignment
    
        return lr_final, hr_final
    
    def rectify_img(self, img, pts, margin=2):
        # obtain a consistent order of the points and unpack them individually
        # rect = order_points(pts)
        (tl, tr, br, bl) = pts
     
        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
     
        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        maxWidth += margin*2
        maxHeight += margin*2
     
        # now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
        ww = maxWidth - 1 - margin
        hh = maxHeight - 1 - margin
        c1 = [margin, margin]
        c2 = [ww, margin]
        c3 = [ww, hh]
        c4 = [margin, hh]

        dst = np.array([c1, c2, c3, c4], dtype = 'float32')
        pts = np.array(pts, dtype='float32')
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
     
        return warped 
    
    def extract_plate_numbers(self, file_path, pattern):
        # List to store extracted plate numbers
        plate_numbers = []
        
        # Open the text file
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Search for the pattern in the current line
                matches = re.search(pattern, line)
                # If a match is found
                if matches:
                    # Extract the matched string
                    plate_number = matches.group(1)
                    # Add the extracted plate number to the list
                    plate_numbers.append(plate_number)
        
        # Return the list of extracted plate numbers
        return plate_numbers[0]
    
    def get_pts(self, file):
        file = file.with_suffix('.json')
        with open(file, 'r') as j:
            pts = json.load(j)['shapes'][0]['points']
        
        return pts
    
    def collate_fn(self, datas):
        lrs = []
        hrs = []
        gts = []
        file_name = []
        
        target_h = self.imgH  # Target height for padding (e.g., self.imgH for your dataset)
        target_w = self.imgW  # Target width for padding (e.g., self.imgW for your dataset)
    
        
        for item in datas: 
            lr_path = random.choice(list(item['imgs'].rglob('lr*.png')))
            img_lr = self.Open_image(lr_path)
            
            hr_path = random.choice(list(item['imgs'].rglob('hr*.png')))
            img_hr = self.Open_image(hr_path)
            gt = self.extract_plate_numbers(next(hr_path.parent.rglob('*.txt')), pattern=r'(\w+)')
            # hr = K.enhance.equalize_clahe(hr, clip_limit=4.0, grid_size=(2, 2))
            
            if self.aug is True:
                # augment = np.random.choice(self.transform, replace = True)
                rectify_assert = random.random()
            else:
                rectify_assert = 1.0
            
            if self.aug:
                if rectify_assert < 0.5:
                    img_lr = self.rectify_img(img_lr, self.get_pts(lr_path), margin=2)
                    img_hr = self.rectify_img(img_hr, self.get_pts(hr_path), margin=2)
                
                # Apply consistent augmentation to both LR and HR images
                img_lr, img_hr = self.augment_images(
                        lr_img=img_lr,
                        hr_img=img_hr,
                        augmentations=self.transform,
                        target_h=target_h,
                        target_w=target_w
                    )
            
            img_lr, _, _, _ = self.pad_with_mask(img_lr, self.ar-0.15, self.ar+0.15, self.background)
            img_lr = resize_fn(img_lr, (self.imgH, self.imgW))
            img_hr = K.enhance.equalize_clahe(transforms.ToTensor()(Image.fromarray(img_hr)).unsqueeze(0), clip_limit=4.0, grid_size=(2, 2))
            img_hr = K.utils.tensor_to_image(img_hr.mul(255.0).byte())
            img_hr, _, _, _ = self.pad_with_mask(img_hr, self.ar-0.15, self.ar+0.15, self.background)  
            img_hr = resize_fn(img_hr, (2*self.imgH, 2*self.imgW))

            lrs.append(img_lr)
            hrs.append(img_hr)
            gts.append(gt)
            
        lr = torch.stack(lrs, dim=0)
        hr = torch.stack(hrs, dim=0)
        
        gt = gts
        del lrs
        del hrs
        del gts
        if self.test and not self.lbp:
            return {
                'lr': lr, 'hr': hr, 'gt': gt, 'name': file_name
                    }
        else:
            return {
                'lr': lr, 'hr': hr, 'gt': gt
                }
    
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
