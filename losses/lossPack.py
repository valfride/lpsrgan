import re
import cv2
import yaml
import torch
import kornia
import losses
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
# from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from PIL import Image
from pathlib import Path
from keras.models import Model
from losses import register, make
from kornia.losses import SSIMLoss
from torchvision import transforms
from torch.autograd import Variable
from memory_profiler import profile
# from matplotlib import pyplot as plt
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from torchvision.transforms.functional import to_pil_image

import gc

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-'+alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            if len(item)<1:
                continue
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def encode_char(self, char):

        return self.dict[char]
    
    def encode_list(self, text, K=7):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.
            K : the max length of texts

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # print(text)
        length = []
        all_result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:
            result = []
            if decode_flag:
                item = item.decode('utf-8','strict')
            # print(item)
            length.append(len(item))
            for i in range(K):
                # print(item)
                if i<len(item): 
                    char = item[i]
                    # print(char)
                    index = self.dict[char]
                    result.append(index)
                else:
                    result.append(0)
            all_result.append(result)
        return (torch.LongTensor(all_result))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
    
    def decode_list(self, t):
        texts = []
        for i in range(t.shape[0]):
            t_item = t[i,:]
            char_list = []
            for i in range(t_item.shape[0]):
                if t_item[i] == 0:
                    pass
                    # char_list.append('-')
                else:
                    char_list.append(self.alphabet[t_item[i]])
                # print(char_list, self.alphabet[44])
            # print('char_list:  ' ,''.join(char_list))
            texts.append(''.join(char_list))
        # print('texts:  ', texts)
        return texts

    def decode_sa(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(text_index):
            text = ''.join([self.alphabet[i] for i in text_index[index, :]])
            texts.append(text.strip('-'))
        return texts


@register('CrossEntropyLoss')
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss = nn.CrossEntropyLoss(weight=self.weight, 
                                        size_average=self.size_average, 
                                        ignore_index=self.ignore_index, 
                                        reduce=self.reduce, 
                                        reduction=self.reduction, 
                                        label_smoothing=self.label_smoothing)
        
    def forward(self, v1, v2):
        return self.loss(v1, v2)
    

@register('ssim_loss')
class sr_loss(nn.Module):
    def __init__(self, window_size=3, reduction='mean', padding='same'):
        super(sr_loss, self).__init__()
        self.window_size = window_size
        self.ssim = SSIMLoss(window_size)
        #previous training, uncomment for future analysis
        #self.MS_SSIMLoss = MS_SSIMLoss(alpha=0.84).cuda()
        # self.MS_SSIMLoss = MS_SSIMLoss(sigmas=[0.5, 1.0, 2.0], data_range=1.0, K=(0.01, 0.03), alpha=0.1, compensation=200.0, reduction='mean').cuda()
    def forward(self, im1, im2):
        return self.ssim(im1, im2) 
    
@register('MaskLoss')
class MaskLoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='none'):
        super(MaskLoss, self).__init__()
        self.loss = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)
    
    def generate_mask(self, num_channels, image_width, image_height):
        # Character range within the image height
        character_range_start = 1 * image_height // 3
        character_range_end = 2 * image_height // 3  
        
        # Create a Gaussian-like weight map
        weight_map = torch.ones(image_height)
        weight_map[:character_range_start] = 0.5
        weight_map[character_range_end:] = 0.5
        
        # Reshape the weight map to match the height of the image
        return weight_map.unsqueeze(0).unsqueeze(-1).repeat(1, num_channels, 1, image_width).cuda()
    
    def forward(self, im1, im2):
        return torch.mean(self.generate_mask(im1.shape[-3], im1.shape[-1], im1.shape[-2]) * self.loss(im1, im2))
           
@register('L1Loss')
class L1Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(size_average=size_average, reduce=reduce, reduction=reduction)
                    
    def forward(self, v1, v2):
        return self.loss(v1, v2)
    
@register('SmoothL1Loss')
class SmoothL1Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(size_average=size_average, reduce=reduce, reduction=reduction, beta=beta)

            
    def forward(self, v1, v2):
        return self.loss(v1, v2)

@register('MSELoss')
class MSELoss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()        
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.loss = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

            
    def forward(self, v1, v2):
        return self.loss(v1, v2)

@register('PSTD_loss')
class PSTD_loss(nn.Module):
    def __init__(self, loss_specs=None):
        super(PSTD_loss, self).__init__()
        self.l1loss = nn.L1Loss()
        
    def compute_gradients(self, input_image):
        input_image_gray = torch.mean(input_image, dim=1, keepdim=True)
        
        padded_image_x = torch.nn.functional.pad(input_image_gray, (0, 1, 0, 0))
        padded_image_y = torch.nn.functional.pad(input_image_gray, (0, 0, 0, 1))
        
        gradient_x = torch.abs(padded_image_x[:, :, :, :-1] - padded_image_x[:, :, :, 1:])
        gradient_y = torch.abs(padded_image_y[:, :, :-1, :] - padded_image_y[:, :, 1:, :])
        
        gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = torch.atan2(gradient_y, gradient_x)
        
        return torch.cat((gradient_magnitude, gradient_direction), dim=1)
    
    def forward(self, im1, im2):
        grad_im1 = self.compute_gradients(im1[0])
        grad_im2 = self.compute_gradients(im2[0])
        
        return self.l1loss(im1[0], im2[0]) + 0.01*self.l1loss(grad_im1, grad_im2) + 0.001*self.l1loss(im1[1], im2[1])


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )
import kornia as K

@register('Perceptual_loss_CG')
class Perceptual_loss_CG(nn.Module):
    def __init__(self, load=None, loss_weight=None, loss1_specs=None, loss2_specs=None, use_external_ocr=False):
        super(Perceptual_loss_CG, self).__init__()
        
        # Create loss functions using provided specifications
        # self.loss1 = make(loss1_specs)
        self.load = load
        self.loss1 = make(loss1_specs)
        self.loss2 = make(loss2_specs)
        # Check if GPUs are available using TensorFlow (TF)
        
        if self.load:
            gpus = tf.config.experimental.list_physical_devices('GPU')
    
            if gpus:
                # Configure GPU memory growth for each GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Limit GPU memory fraction for each GPU (here, 100% of GPU memory)
                gpu_fraction = 0.01
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_fraction * 12288))]
                    )
            
            # Initialize attributes related to loading an OCR model
            self.load = Path(load)
            self.OCR = load_model(self.load.as_posix())
            self.OCR = Model(self.OCR.input, self.OCR.layers[-41].output)
            self.IMAGE_DIMS = self.OCR.layers[0].input_shape[0][1:]
            self.parameters = np.load(self.load.as_posix() + '/parameters.npy', allow_pickle=True).item()
            self.tasks = self.parameters['tasks']
            self.ocr_classes = self.parameters['ocr_classes']
            self.num_classes = self.parameters['num_classes']
            #change for testing
            #remenber to change back to True Valfride
            self.padding = False
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
            self.r_padding = False
            self.background = (127, 127, 127)
            
    def OCR_pred(self, img, fl = None, convert_to_bgr=True):
        if convert_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        if self.padding:
            img, _, _ = padding(img, self.min_ratio, self.max_ratio, color = (127, 127, 127))        
        
        img = cv2.resize(img, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))        
        img = img_to_array(img)        
        img = img.reshape(1, self.IMAGE_DIMS[0], self.IMAGE_DIMS[1], self.IMAGE_DIMS[2])
        
        img = (img/255.0).astype('float')
        predictions = self.OCR(img, training=False)
        # _ = gc.collect()      
        return predictions
        
    def get_logits(self, images):
        batch = []
        
        for img in images:
            img = np.array(img.detach().cpu().permute(1, 2, 0)*255).astype('uint8')

            if self.OCR is not None:
                features = self.OCR_pred(img)
            batch.append(features)
        logits = Variable(torch.as_tensor(np.array(batch)), requires_grad=True)
        # logits = logits.view(logits.size(0), -1)
        
        return logits
    
    def forward(self, im1, im2):
        if self.r_padding is True:
            im1 = K.utils.tensor_to_image(im1.mul(255.0).byte())
            im2 = K.utils.tensor_to_image(im2.mul(255.0).byte())
            im1, _, _ = padding(im1, self.min_ratio, self.max_ratio, self.background)
            im2, _, _ = padding(im2, self.min_ratio, self.max_ratio, self.background)
            im1 = resize_fn(im1, (32, 96)).unsqueeze(0)
            im2 = resize_fn(im2, (32, 96)).unsqueeze(0)
            # resize_fn(lr, (self.imgH, self.imgW))
        log1, log2 = self.get_logits(im1), self.get_logits(im2)
        loss1 = self.loss1(log1, log2)
        loss2 = self.loss2(im1, im2)
        
        return loss1 + loss2

@register('OCR_perceptual_loss')
class OCR_perceptual_loss(nn.Module):
    def __init__(self, load=None, loss_weight=None, loss1_specs=None, loss2_specs=None, use_external_ocr=False):
        super(OCR_perceptual_loss, self).__init__()
        
        # Create loss functions using provided specifications
        # self.loss1 = make(loss1_specs)
        self.load = load
        self.loss1 = CrossEntropyLoss()
        self.loss2 = make(loss2_specs)
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        # "皖南沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789 "
        self.converter = strLabelConverter(self.alphabet)
        # Set the weight for the perceptual loss
        self.weight = loss_weight
        # Check if GPUs are available using TensorFlow (TF)
        
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
            self.padding = True
            self.aspect_ratio = self.IMAGE_DIMS[1]/self.IMAGE_DIMS[0]
            self.min_ratio = self.aspect_ratio - 0.15
            self.max_ratio = self.aspect_ratio + 0.15
            self.OCR.compile()
    
    def layout_penalty(self, pred_layout, gt_layout):
        penalty = 0
        for pred_char, gt_char in zip(pred_layout, gt_layout):
            # Check if a number is predicted instead of a letter
            if pred_char.isdigit() and gt_char.isalpha():
                penalty += 0.4
            # Check if a letter is predicted instead of a number
            elif pred_char.isalpha() and gt_char.isdigit():
                penalty += 0.5
        return penalty
    
    def OCR_pred(self, img, convert_to_bgr=True):
        preds = []
        imgs = []
        
        # if convert_to_bgr:
        #     for i, im in enumerate(img):
        #         im = np.array(im.detach().cpu().permute(1, 2, 0)*255).astype('uint8')
        #         img[i] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
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
    
    def one_hot_encode(self, indices, num_classes):
        return F.one_hot(indices, num_classes=num_classes).float()
    
    def visually_similar_penalty(self, pred_char, gt_char, confusing_pairs):
        # confusing_pairs = [("M", "W"), ("H", "M"),
        #                    ("H", "W"), ("Q", "O"),
        #                    ("Y", "V"), ("C", "G"),
        #                    ("9", "8"), ("5", "9")]
        for pair in confusing_pairs:
            if pred_char in pair and gt_char in pair:
                return 1
        return 0
    
    def custom_cross_entropy(self, pred_one_hot, gt_one_hot, weights=None):
        if weights is not None:
            ce_loss = F.cross_entropy(pred_one_hot, gt_one_hot, weight=weights)
        else:
            ce_loss = F.cross_entropy(pred_one_hot, gt_one_hot)
        return ce_loss
    
    def forward(self, im1, im2, predsSR, gt, confusing_pairs, alpha=1.0):
        if self.load and predsSR is None:
            pred1_layout, pred2_layout = self.OCR_pred(im1, convert_to_bgr=True), gt
            # self.OCR_pred(im2, convert_to_bgr=True)
        else:
            pred1_layout, pred2_layout = predsSR, gt
        penalty = 0
        for pred, corr in zip(pred1_layout, pred2_layout):
            penalty += self.layout_penalty(pred, corr)
        
        penalty = penalty/len(pred1_layout)
        
        pred1, pred2 = self.converter.encode_list(pred1_layout, K=7).cuda(), self.converter.encode_list(pred2_layout, K=7).cuda()
        pred1 = self.one_hot_encode(pred1, len(self.alphabet))
        
        weights = torch.ones((pred1.shape[0], pred1.shape[2]))
        
        for i, (plate1, plate_gt) in enumerate(zip(pred1_layout, pred2_layout)):
            for char1, char_gt in zip(plate1, plate_gt):
                if self.visually_similar_penalty(char1, char_gt, confusing_pairs):
                    idx = self.converter.encode_char(char_gt)
                    weights[i][idx] += 0.5
        
        pred1 = torch.chunk(pred1, pred1.size(0), 0)
        # print(weights)
        loss1 = 0
        for (i, item) in enumerate(pred1):
            item = item.squeeze().cuda()
            gt = pred2[i,:].cuda()
            loss_item = self.custom_cross_entropy(item, gt, weights=weights[i].cuda())
            loss1 += loss_item
        loss1 = loss1/(i+1)
        loss2 = self.loss2(im1, im2)
     
        return loss1 + loss2 + alpha*penalty
    
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


class RunningAverage:
    def __init__(self, alpha=0.99):
        self.alpha = alpha  # Smoothing factor for EMA
        self.value = None   # Running average value

    def update(self, new_value):
        if self.value is None:
            self.value = new_value  # Initialize on first update
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value  # Update EMA

    def get(self):
        return self.value

@register('SROCR_loss')
class SROCR_loss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        super(SROCR_loss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.loss = nn.CrossEntropyLoss(weight=self.weight, 
                                        size_average=self.size_average, 
                                        ignore_index=self.ignore_index, 
                                        reduce=self.reduce, 
                                        reduction=self.reduction, 
                                        label_smoothing=self.label_smoothing)
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.converter = strLabelConverter(self.alphabet)
        self.loss_SR = L1Loss()
        
        self.running_avg_OCR = RunningAverage(alpha=0.99)
        self.running_avg_SR = RunningAverage(alpha=0.99)
        
    def forward(self, img1, img2, preds, lp_gt):
        text = self.converter.encode_list(lp_gt, K=7).cuda()
        preds = torch.chunk(preds, preds.size(0), 0)
        loss_ocr = 0
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = self.loss(item, gt)
            loss_ocr += loss_item
        loss_ocr = loss_ocr/(i+1)
        loss_sr = self.loss_SR(img1, img2)
        
        self.running_avg_OCR.update(loss_ocr.detach())
        self.running_avg_SR.update(loss_sr.detach())
        
        normalized_loss_OCR = loss_ocr / self.running_avg_OCR.get()
        normalized_loss_SR = loss_sr / self.running_avg_SR.get()
        
        # print(f"loss_ocr: {0.5*loss_ocr} --- loss_sr: {loss_sr}")
        
        # return 0.5*loss_ocr + loss_sr
        return normalized_loss_OCR + normalized_loss_SR 


import hyperlpr3 as lpr3
from torchvision.models import vgg19

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
        
        total_loss = (
            self.weights[0] * loss_mae +
            self.weights[1] * vgg_loss +
            self.weights[2] * loss_adv +
            self.weights[3] * ocr_loss            
            )
        
        return loss_mae + vgg_loss + ocr_loss, preds


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features.eval()
        
        # Block indices for the last conv layer in each VGG block
        self.block_indices = [3, 8, 17, 26, 35]  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
        # self.blocks = nn.ModuleList([self.vgg[:idx+1] for idx in self.block_indices])
        
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
        
        # if convert_to_bgr:
        #     for i, im in enumerate(img):
        #         im = np.array(im.detach().cpu().permute(1, 2, 0)*255).astype('uint8')
        #         img[i] = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
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
        predicted_text_hr = self.OCR_pred(hr_image)
        
        for i in range(batch_size):
            # print(predicted_text_hr[i], predicted_text[i], label_texts[i])

            if predicted_text[i] != label_texts[i]:
                total_loss += torch.tensor(1.0, device=sr_image.device)

        # Average loss over the batch
        return total_loss / batch_size, predicted_text

