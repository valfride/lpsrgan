
import re
import gc
import cv2
import yaml
import torch
import kornia
import losses
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F


from PIL import Image
from pathlib import Path
from keras.models import Model
from losses import register, make
from kornia.losses import SSIMLoss
from torchvision import transforms
from torch.autograd import Variable
from memory_profiler import profile
from torchvision.models import vgg19
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from skimage.metrics import structural_similarity

def load_model(path):
	with open(path + '/model.json', 'r') as f:
		json = f.read()

	model = model_from_json(json)
	model.load_weights(path + '/weights.hdf5')
        
	return model

def padding(img, min_ratio, max_ratio, color = (0, 0, 0)):
    # Get the height and width of the input image
	img_h, img_w = np.shape(img)[:2]

    
    # Initialize variables for border width and height
	border_w = 0
	border_h = 0

    # Calculate the aspect ratio (width divided by height) of the input image
	ar = float(img_w)/img_h

    # Check if the aspect ratio is within the specified range [min_ratio, max_ratio]
	if ar >= min_ratio and ar <= max_ratio:
		return img, border_w, border_h

    # If the aspect ratio is less than the minimum allowed ratio (min_ratio)
	if ar < min_ratio:
		while ar < min_ratio:
			border_w += 1 # Increase the border width
			ar = float(img_w+border_w)/(img_h+border_h)

    # If the aspect ratio is greater than the maximum allowed ratio (max_ratio)
	else:
		while ar > max_ratio:
			border_h += 1
			ar = float(img_w)/(img_h+border_h)

    # Calculate half of the border width and height
	border_w = border_w//2
	border_h = border_h//2

    # Use OpenCV's copyMakeBorder function to add padding to the image
	img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
    
	return img, border_w, border_h  

@register('lpsrganLoss')
class LPSRGANLoss(nn.Module):
    def __init__(self, load=None, weight=None, size_average=None):
        super(LPSRGANLoss, self).__init__()  # Corrected super() call
        self.weights = [0.01, 1.0, 0.05, 0.05]
        self.mae = nn.L1Loss()
        self.vgg_loss = VGG19PerceptualLoss()
        self.ocr_loss = OCRLoss(load)
          # Uncommented and initialized

    def forward(self, img1, img2, lp_gt, loss_adv):
        # Ensure img1 and img2 are in the correct format
        loss_mae = self.mae(img1, img2)  # Corrected L1Loss usage
        vgg_loss = self.vgg_loss(img1, img2)
        ocr_loss, preds = self.ocr_loss(img1, img2, lp_gt)
        
        # total_loss = (
        #     self.weights[0] * loss_mae +
        #     self.weights[1] * vgg_loss +
        #     self.weights[2] * loss_adv +
        #     self.weights[3] * ocr_loss            
        #     )
        
        return loss_mae + vgg_loss + ocr_loss, preds


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features.eval()
        
        # Block indices for the last conv layer in each VGG block
        self.block_indices = [3, 8, 17, 26, 35]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
        
        blocks = []
        last = 0
        for actual in self.block_indices:
            blocks.append(self.vgg[last:actual+1])
            last = actual+1
        self.blocks = nn.ModuleList(blocks)
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Normalize input: [0, 1] → VGG19 expects ImageNet stats
        x = (x - 0.45) / 0.225  # Simplified normalization
        
        features = []
        for block in self.blocks:
            # print(block)
            x = block(x)
            features.append(x)
        return features

class VGG19PerceptualLoss(nn.Module):
    def __init__(self, weights=[0.1, 0.1, 1.0, 1.0, 1.0]):
        super().__init__()
        self.vgg = VGG19FeatureExtractor().cuda()
        self.l1 = nn.L1Loss()
        self.weights = weights  # [p1, p2, p3, p4, p5]

    def forward(self, sr, hr):
        # Get features for SR and HR images
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)  # Detach HR to avoid backprop
        
        # Compute weighted L1 loss for each block
        loss = 0.0
        for i, (sr_f, hr_f) in enumerate(zip(sr_features, hr_features)):
            loss += self.weights[i] * self.l1(sr_f, hr_f)
        return loss


class OCRLoss(nn.Module):
    def __init__(self, load=None):
        super(OCRLoss, self).__init__()
        # self.ocr = lpr3.LicensePlateCatcher()  # Initialize OCR model
        self.load = load
        
        if self.load:
            gpus = tf.config.experimental.list_physical_devices('GPU')
    
            if gpus:
                # Configure GPU memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Limit GPU memory fraction for each GPU (here, 100% of GPU memory)
                gpu_fraction = 0.2
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_fraction * 12288))]
                    )
            
            # Initialize attributes related to loading an OCR model
            self.load = Path(load)
            self.OCR = load_model(self.load.as_posix())
            # self.OCR = Model(self.OCR.input, self.OCR.layers[-41].output)
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(self.load.as_posix() + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            #change for testing
            #remenber to change back to True Valfride
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
            self.r_padding = False
            self.background = (127, 127, 127)
            
        else:
            print("OCRLoss :: No valid LPR-OCR!!!")
            
    def OCR_pred(self, img, convert_to_bgr=True):
        preds = []
        imgs = []
        
        if self.padding and convert_to_bgr:
            for i, im in enumerate(img):
                im = np.array(im.detach().cpu().permute(1, 2, 0)*255).astype('uint8')
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                im , _, _ = padding(im, self.min_ratio, self.max_ratio, color = (127, 127, 127))
                imgs.append(im)
                
        for im in imgs:
            im = cv2.resize(im, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
            im = img_to_array(im)        
            im = im.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
            
            im = (im/255.0).astype('float')
            predictions = self.OCR(im, training=False)
    
            plates = [''] * 1
            for task_idx, pp in enumerate(predictions):
                idx_aux = task_idx + 1
                task = self.tasks[task_idx]
    
                if re.match(r'^char[1-9]$', task):
                    for aux, p in enumerate(pp):
                        plates[aux] += self.ocr_classes['char{}'.format(idx_aux)][np.argmax(p)]
                else:
                    raise Exception('unknown task \'{}\'!'.format(task))
            preds.extend(plates)   
        return preds    
    
    def forward(self, sr_image, hr_image, label_texts):
        """
        Args:
            sr_image: Tensor of shape [batch_size, channels, height, width]
            label_texts: List of strings (e.g., ["粤A12345", "粤B5678", ...])
        """
        batch_size = sr_image.size(0)
        total_loss = torch.tensor(0.0, device=sr_image.device)
        predicted_text = self.OCR_pred(sr_image)
        
        for i in range(batch_size):
            if predicted_text[i] != label_texts[i]:
                total_loss += torch.tensor(1.0, device=sr_image.device)

        # Average loss over the batch
        return total_loss / batch_size, predicted_text

