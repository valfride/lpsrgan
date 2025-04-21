import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from train_funcs import register
from torchvision import transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim import SGD, Adam
import copy
alpha=1.0
debug = False

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(f'Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, '
          f'Max Allocated: {max_allocated:.2f} MB, Max Reserved: {max_reserved:.2f} MB')

class MemoryProfiler:
    def __init__(self, model):
        self.model = model
        self.forward_hooks = []
        self.backward_hooks = []

    def register_hooks(self):
        for layer in self.model.children():
            forward_hook = layer.register_forward_hook(self.forward_hook_fn)
            backward_hook = layer.register_backward_hook(self.backward_hook_fn)
            self.forward_hooks.append(forward_hook)
            self.backward_hooks.append(backward_hook)

    def forward_hook_fn(self, module, input, output):
        print(f'Forward pass - Layer: {module.__class__.__name__}')
        print_gpu_memory_usage()

    def backward_hook_fn(self, module, grad_input, grad_output):
        print(f'Backward pass - Layer: {module.__class__.__name__}')
        print_gpu_memory_usage()

    def remove_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()
        for hook in self.backward_hooks:
            hook.remove()

def save_visualized_images(image1, image2, image3, output_path, sr_text="dummy", gt_text="dummy"):
    # Get the size of the SR image
    sr_size = image2.size

    # Resize the LR and GT images to match the size of the SR image
    image1_resized = image1.resize(sr_size)
    image3_resized = image3.resize(sr_size)

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the resized LR image with "LR" as the title
    axes[0].imshow(image1_resized)
    axes[0].set_title("LR")
    axes[0].axis('off')

    # Display the SR image with "SR" as the title and the predicted LP text
    axes[1].imshow(image2)
    axes[1].set_title("SR")
    axes[1].axis('off')
    axes[1].text(0.5, -0.1, f"Pred: {sr_text}", transform=axes[1].transAxes, 
                 fontsize=12, ha='center', va='top', color='blue')

    # Display the resized GT image with "GT" as the title and the ground truth LP text
    axes[2].imshow(image3_resized)
    axes[2].set_title("GT")
    axes[2].axis('off')
    axes[2].text(0.5, -0.1, f"GT: {gt_text}", transform=axes[2].transAxes, 
                 fontsize=12, ha='center', va='top', color='green')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')  # Ensure the text is not cut off
    plt.close(fig)
    

    
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

@register('PARALLEL_TRAINING')
def train_parallel(train_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, ocr_opt, sr_opt, confusing_pair, *args):
    config = args[0]
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    debug = config.get('debug', False)
    # count=0
    if debug:
        profiler = MemoryProfiler(sr_model)
        profiler.register_hooks()
    
    # Set train mode for models
    ocr_model.train() if not config['MODEL_OCR']['path'] else ocr_model.eval()
    sr_model.train()

    train_loss = []
    pbar = tqdm(train_loader, leave=False, desc='Train')
    
    # Iterate through batches in the training data
    for idx, batch in enumerate(pbar):
        lr_images = batch['lr'].cuda()
        hr_images = batch['hr'].cuda()
        
        text = converter.encode_list(batch['gt'], K=7).cuda()
        _, preds,_ = ocr_model(hr_images)    
        loss_ocr_real = 0
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = ocr_loss_fn(item, gt.cuda())
            loss_ocr_real += loss_item
        loss_ocr_real = loss_ocr_real/(i+1)
        
        # Predict on SR images fake images
        
        sr = sr_model(lr_images)
        _, preds,_ = ocr_model(sr.detach())
        preds_all_sr = preds
        _, preds_all_sr = preds_all_sr.max(2)
        sim_preds_sr = converter.decode_list(preds_all_sr)
        loss_ocr_fake = 0
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = ocr_loss_fn(item, gt.cuda())
            loss_ocr_fake += loss_item
        loss_ocr_fake = loss_ocr_fake/(i+1)
        
        # Calculate and backward loss on OCR
        # loss_ocr = alpha*(1/torch.log(loss_ocr_fake + 1e-6)) + loss_ocr_real
        # loss_ocr = alpha*torch.exp(-loss_ocr_fake + 1e-6) + loss_ocr_real
        loss_ocr = loss_ocr_fake + loss_ocr_real
        
        if config['MODEL_OCR']['path'] is None and config['MODEL_OCR']['lock'] is True:
            ocr_opt.zero_grad()
            loss_ocr.backward()
            ocr_opt.step()
        # else:
        #     loss_ocr.detach()
        # Predict on HR images ground truth
        
        # _, preds,_ = ocr_model(hr_images)
        # loss_ocr_real = 0
        # preds = torch.chunk(preds, preds.size(0), 0)
        # for (i, item) in enumerate(preds):
        #     item = item.squeeze()
        #     gt = text[i,:]
        #     loss_item = ocr_loss_fn(item, gt)
        #     loss_ocr_real += loss_item
        # loss_ocr_real = loss_ocr_real/(i+1)

        # Predict on SR images Fake Images
        
        # sr = sr_model(lr_images)
        # _, preds,_ = ocr_model(sr.detach())   
        # preds_all_sr = preds
        
        if not config['CM']:
            loss_ocr_fake = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = F.cross_entropy(item, gt)
                loss_ocr_fake += loss_item
            loss_ocr_fake = loss_ocr_fake/(i+1)
        
        
        # _, preds_all_sr = preds_all_sr.max(2)
        # sim_preds_sr = converter.decode_list(preds_all_sr)
        # Calculate and backward loss on SR_NET
        
        # loss_sr = sr_loss_fn(sr, batch['hr'].cuda()) + loss_ocr_fake + 1/torch.log((loss_ocr_real + 1e-6))
        # NEW
        if config['CM']:
            loss_sr_1 = sr_loss_fn(sr, hr_images, sim_preds_sr, batch['gt'], confusing_pair)
        else:
            loss_sr_1 = sr_loss_fn(sr, hr_images) + loss_ocr_fake
        loss_sr =  loss_sr_1
        # OLD
        # loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda(), sim_preds_sr, batch['gt'], confusing_pair)
        
        sr_opt.zero_grad()
        loss_sr.backward()
        sr_opt.step()
        
        if debug:
            profiler.remove_hooks()
        
        if idx%3 == 0:
            rand_img = random.randint(0, len(batch['lr'])-1)
            image1 = transforms.ToPILImage()(batch['lr'][rand_img].to('cpu'))
            image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
            image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
            save_visualized_images(image1, image2, image3, config['MODEL_SR']['name']+config['tag_view']+'.png')
            
        train_loss.append(loss_sr.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss),
                          'loss_ocr': loss_ocr.detach().item(),
                          'loss_sr': loss_sr_1.detach().item(),
                          'loss_fake': loss_ocr_fake.detach().item(),
                          'loss_real': loss_ocr_real.detach().item()})
        # count+=1
        # if count>10:
        #     break
        # break
    return sum(train_loss)/len(train_loss)
        
@register('PARALLEL_VALIDATION')
def validation_parallel(val_loader, ocr_model, sr_model, ocr_loss_fn, sr_loss_fn, confusing_pairs, *args):
    #debuggin purpose
    # count=0
    ###
    config = args[0]
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    
    #set train mode for models
    ocr_model.eval()
    sr_model.eval()

    val_loss = []
    
    #create a progress bar for training (tqdm library)
    pbar = tqdm(val_loader, leave=False, desc='Val')
    n_correct = 0
    total = 0
    
    n_correct_sr = 0
    total_sr = 0
    
    if config['CM']:
        preds_sr_cm = []
        preds_gt_cm = []
    # Disable gradient computation during validation
    with torch.no_grad():
        
        # Iterate through batches in the training data
        
        for idx, batch in enumerate(pbar):
            text = converter.encode_list(batch['gt'], K=7).cuda()
            
            # Predict on HR images ground truth
            
            _, preds,_ = ocr_model(batch['hr'].cuda())
            preds_all=preds     
            loss_ocr_real = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = ocr_loss_fn(item, gt)
                loss_ocr_real += loss_item
            loss_ocr_real = loss_ocr_real/(i+1)
            
            _, preds_all = preds_all.max(2)
            sim_preds = converter.decode_list(preds_all)
            text_label = batch['gt']
            
            for pred, target in zip(sim_preds, text_label):
                pred = pred.replace('-', '')
                if pred == target:
                    n_correct += 1
                total += 1
            
            # Predict on SR images fake images
            
            sr = sr_model(batch['lr'].cuda())
            _, preds,_ = ocr_model(sr.detach())
            loss_ocr_fake = 0
            preds = torch.chunk(preds, preds.size(0), 0)
            for (i, item) in enumerate(preds):
                item = item.squeeze()
                gt = text[i,:]
                loss_item = ocr_loss_fn(item, gt)
                loss_ocr_fake += loss_item
            loss_ocr_fake = loss_ocr_fake/(i+1)
    
            # Calculate loss on OCR
            
            # loss_ocr = alpha*(1/torch.log(loss_ocr_fake + 1e-6)) + loss_ocr_real
            # loss_ocr = alpha*torch.exp(-loss_ocr_fake + 1e-6) + loss_ocr_real
            loss_ocr = loss_ocr_fake + loss_ocr_real
            # Predict on SR images fake images
            
            sr = sr_model(batch['lr'].cuda())
            _, preds,_ = ocr_model(sr.detach())  
            preds_all_sr = preds        
            
            if not config['CM']:
                loss_ocr_fake = 0
                preds = torch.chunk(preds, preds.size(0), 0)
                for (i, item) in enumerate(preds):
                    item = item.squeeze()
                    gt = text[i,:]
                    loss_item = F.cross_entropy(item, gt)
                    loss_ocr_fake += loss_item
                loss_ocr_fake = loss_ocr_fake/(i+1)
                
            _, preds_all_sr = preds_all_sr.max(2)
            
            #create lists for confusion matrix
            if config['CM']:
                preds_sr_cm.extend(preds_all_sr.detach().cpu().numpy())
                preds_gt_cm.extend(converter.encode_list(batch['gt']).detach().cpu().numpy())
            
            
            sim_preds_sr = converter.decode_list(preds_all_sr)
            text_label = batch['gt']
            
            for pred, target in zip(sim_preds_sr, text_label):
                pred = pred.replace('-', '')
                if pred == target:
                    n_correct_sr += 1
                total_sr += 1
            
            #NEW
            if config['CM']:
                loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda(), sim_preds_sr, batch['gt'], confusing_pairs)
            else:
                loss_sr_1 = sr_loss_fn(sr, batch['hr'].cuda()) + loss_ocr_fake
            loss_sr =  loss_sr_1
            # OLD
            # loss_sr = sr_loss_fn(sr, batch['hr'].cuda(), sim_preds_sr,  batch['gt'], confusing_pairs)# + loss_ocr_fake + torch.exp(-loss_ocr_real + 1e-6)
            # loss_sr = sr_loss_fn(sr, batch['hr'].cuda()) + loss_ocr_fake + 1/torch.log((loss_ocr_real + 1e-6))

            if idx%3 == 0:
                rand_img = random.randint(0, len(batch['lr'])-1)
                image1 = transforms.ToPILImage()(batch['lr'][rand_img].to('cpu'))
                image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['MODEL_SR']['name']+config['tag_view']+'.png')
            
            val_loss.append((loss_sr.detach().item()+loss_ocr.detach().item())/2)
            pbar.set_postfix({'loss': sum(val_loss)/len(val_loss), 'loss_fake': loss_ocr_fake.detach().item(), 'loss_real': loss_ocr_real.detach().item(), 'loss_ocr': loss_ocr.detach().item()})
            
            
            # count+=1
            # if count>10:
            #     break
            # break
        
        if config['CM']:
            flattened_preds = np.concatenate(preds_sr_cm)
            flattened_gts = np.concatenate(preds_gt_cm)
        
            conf_matrix = confusion_matrix(flattened_preds, flattened_gts, labels=range(len(alphabet)))
            conf_matrix_normalized = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            confusing_pairs = extract_confusing_pairs(conf_matrix_normalized, alphabet, 0.25)
            print(confusing_pairs)
        else:
            confusing_pairs = []
        
        print("\nAccuracy for HR")
        for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
        # accuracy = (n_correct / len(val_loader))
        accuracy = (n_correct / float(total))
        print(f'accuracy: {accuracy*100:.2f}%')
        
        print("Accuracy for SR")
        for raw_pred, pred, gt in zip(preds_all_sr, sim_preds_sr, text_label):
            raw_pred = raw_pred.data
            pred = pred.replace('-', '')
            print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
        # accuracy = (n_correct / len(val_loader))
        accuracy_sr = (n_correct_sr / float(total_sr))
        print(f'accuracy: {accuracy_sr*100:.2f}%')
        
        # print("VAL ACCURACY: ", accuracy_sr, 1-accuracy_sr)
        
        return sum(val_loss)/len(val_loss), 1-accuracy_sr, confusing_pairs

def extract_confusing_pairs(conf_matrix, class_names, threshold=10):
    confusing_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i, j] > threshold:
                confusing_pairs.append((class_names[i], class_names[j]))
    return confusing_pairs

@register('GP_LPR_TRAIN')
def train_ocr(train_loader, model, opt, loss_fn, confusing_pairs, *args):
    config = args[0]
    converter = strLabelConverter(config['alphabet'])
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    pbar = tqdm(train_loader, leave=False, desc='train')
    train_loss = []
    
    for i_batch, batch in enumerate(pbar):
        text = converter.encode_list(batch['text'], K=7).cuda()
        _, preds,_ = model(batch['img'].cuda())
        loss = 0        
        preds = torch.chunk(preds, preds.size(0), 0)
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss_item = loss_fn(item, gt)
            loss+= loss_item
        loss = loss/(i+1)
        
        
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss.append(loss.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss)})
        
    return sum(train_loss)/len(train_loss)

@register('GP_LPR_VAL')
def validation_ocr(val_loader, model, loss_fn, confusing_pairs, *args):
    config = args[0]
    converter = strLabelConverter(config['alphabet'])
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    i = 0
    n_correct = 0
    pbar = tqdm(val_loader, leave=False, desc='val')
    val_loss = []
    total = 0
    for i_batch, batch in enumerate(pbar):
        text = converter.encode_list(batch['text'], K=7).cuda()
        _, preds,_ = model(batch['img'].cuda())
        preds_all = preds        
        
        loss = 0
        preds = torch.chunk(preds, preds.size(0), 0)    
        
        for (i, item) in enumerate(preds):
            item = item.squeeze()
            gt = text[i,:]
            loss += loss_fn(item, gt)
        loss = loss/(i+1)
        
        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = batch['text']
        val_loss.append(loss.detach().item())
        
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace('-', '')
            if pred == target:
                n_correct += 1
            total += 1
        
        pbar.set_postfix({'loss': sum(val_loss)/len(val_loss)})  
    
    print()
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))    
    # accuracy = (n_correct / len(val_loader))
    accuracy = (n_correct / float(total))
    print(f'accuracy: {accuracy*100:.2f}%')
    return 1-accuracy, None

@register('SR_TRAIN')
def train_sr(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    # Set the model to training mode
    model.train()
    # count=0
    train_loss = []
    
    # Zero out the gradients in the optimizer
    optimizer.zero_grad()
    
    # Create a progress bar for training (using tqdm library)
    pbar = tqdm(train_loader, leave=False, desc='train')
    
    # Iterate through batches in the training data
    for idx, batch in enumerate(pbar):
        # Forward pass: Generate super-resolution (sr) images using the model
        # model = model.cpu()
        sr = model(batch['lr'].cuda())
        # Calculate the loss between the generated sr images and the high-resolution (hr) ground truth images
        if config['CM']:
            loss = loss_fn(sr, batch['hr'].cuda(), None, batch['gt'], confusing_pair)      
        else:
            loss = loss_fn(sr, batch['hr'].cuda())        
        
        # Uncomment the following lines if you wish to visualize a random sample.

        if isinstance(sr, tuple):
            Image.fromarray(np.array(transforms.ToPILImage()(sr[0][random.randint(0, len(batch['lr'])-1)].to('cpu'))).astype('uint8')).save(config['model']['name']+'.jpg')
        else:
            if idx%3 == 0:
                rand_img = random.randint(0, len(batch['lr'])-1)
                if batch['lr'][rand_img].dim() > 3:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                else:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png')
        
        
        optimizer.zero_grad() # Zero out the gradients in the optimizer (clear previous gradients)
        loss.backward() # Backpropagation: Compute gradients based on the loss
        optimizer.step() # Update the model's parameters using the optimizer
        
        # Append the current loss to the list of training losses
        train_loss.append(loss.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss)})
        # Update the progress bar to display the running loss
        
        # Clear variables to save memory (optional)
        loss=None
        sr=None
        # count+=1
        
        # if count>=10:
        # break
    # Calculate the average training loss and return it
    return sum(train_loss)/len(train_loss)

def lpsrgan_pre_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    
    model[0].train()
    validation_interval = 20
    criterion = nn.L1Loss() 
    best_psnr = 0.0
    optimizer[0].zero_grad()
    optimizer[1].zero_grad()
    
    
    for g in optimizer[0].param_groups:
        g['lr'] = 2e-4
    
    for iteration in range(1000):  # Adjust to match paper’s 1000K steps
        pbar = tqdm(train_loader, leave=False, desc='train')
        for idx, batch in enumerate(pbar):
            sr_batch = model[0](batch['lr'].cuda())
            
            loss = criterion(sr_batch, batch['hr'].cuda())
            
            optimizer[0].zero_grad()
            loss.backward()
            optimizer[0].step()
            
            pbar.set_postfix({'Pre_train loss': loss.detach().item(), 'Iteration': iteration+1, 'best psnr': best_psnr})
            
            if idx%3 == 0:
                rand_img = random.randint(0, (len(batch['lr'])-1)//2)
                if batch['lr'][rand_img].dim() > 3:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                else:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                image2 = transforms.ToPILImage()(sr_batch[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text='PreTrain', gt_text=batch['gt'][rand_img])
        if iteration % validation_interval == 0:
           best_model_state, best_psnr = lpsrgan_pre_train_val(model[0], best_psnr, config)
           
    optimizer[0].param_groups[0]['lr'] = 1e-4      
    return best_model_state
        
def lpsrgan_pre_train_val(model, best_psnr, config):
    model.eval()
    val_loader = config['val_loader']
    criterion = nn.L1Loss() 
    best_psnr = 0.0
    psnrs = []
    
    def calculate_psnr(sr, hr, max_val=1.0):
        mse = torch.mean((sr - hr) ** 2)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()    
    
    pbar = tqdm(val_loader, leave=False, desc='val')
    for idx, batch in enumerate(pbar):
        sr_batch = model(batch['lr'].cuda())
        loss = criterion(sr_batch, batch['hr'].cuda())
        # print(sr_batch.shape)
        
        if idx%3 == 0:
            rand_img = random.randint(0, (len(batch['lr'])-1)//2)
            if batch['lr'][rand_img].dim() > 3:
                image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
            else:
                image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
            image2 = transforms.ToPILImage()(sr_batch[rand_img].detach().to('cpu'))
            image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
            save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text="PreTrain", gt_text=batch['gt'][rand_img])
        
        psnr = calculate_psnr(sr_batch, batch['hr'].cuda())
        psnrs.append(psnr)
        
        pbar.set_postfix({'Pre_train val loss': loss.detach().item(), 'val_psnr': sum(psnrs)/len(psnrs)})
    
    psnr = sum(psnrs)/len(psnrs)
    if psnr > best_psnr:
        best_psnr = psnr
        # Deep copy the generator's state_dict to a variable
        best_model_state = copy.deepcopy(model.state_dict())
        return best_model_state, best_psnr
    else:
        return None, psnr 
        
    
        
@register('lpsrgan_train')
def lpsrgan_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    model[0].train()
    model[1].train()
    
    train_loss = []
    
    optimizer[0].zero_grad()
    optimizer[1].zero_grad()
    
    if config['epoch'] <= 1:
        best_model_state = lpsrgan_pre_train(train_loader, model, optimizer, loss_fn, confusing_pair, *args)
        
        if best_model_state is not None:
            model[0].load_state_dict(best_model_state)

    
                
    pbar = tqdm(train_loader, leave=False, desc='train')
    for idx, batch in enumerate(pbar):
        sr_batch = model[0](batch['lr'].cuda())
        real_pred = model[1](batch['hr'].cuda())
        fake_pred = model[1](sr_batch.detach())
        loss_d = torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2)
        
        optimizer[1].zero_grad()
        loss_d.backward()
        optimizer[1].step()
        
        sr_batch = model[0](batch['lr'].cuda())
        fake_pred_for_g = torch.mean(model[1](sr_batch))
        loss_adv = torch.mean((fake_pred_for_g - 1)**2)
        loss_g, preds = loss_fn(sr_batch, batch['hr'].cuda(), batch['gt'], loss_adv)
        
        optimizer[0].zero_grad()
        loss_g.backward()
        optimizer[0].step()
        
        train_loss.append(loss_g.detach().item())
        
        if idx%3 == 0:
            rand_img = random.randint(0, (len(batch['lr'])-1)//2)
            if batch['lr'][rand_img].dim() > 3:
                image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
            else:
                image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
            image2 = transforms.ToPILImage()(sr_batch[rand_img].detach().to('cpu'))
            image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
            save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text=preds[rand_img], gt_text=batch['gt'][rand_img])

        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss)})
    return sum(train_loss)/len(train_loss)

@register('lpsrgan_val')
def lpsrgan_val(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    model[0].eval()
    model[1].eval()
    
    val_loss = []
    n_correct=0
    total = 0
    
    pbar = tqdm(val_loader, leave=False, desc='val')
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            sr_batch = model[0](batch['lr'].cuda())
            
            # Update Discriminator
            real_pred = model[1](batch['hr'].cuda())
            fake_pred = model[1](sr_batch.detach())
            loss_d = torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2)
            
            # Update Generator
            loss_g, preds = loss_fn(sr_batch, batch['hr'].cuda(), batch['gt'], loss_d.detach())
            val_loss.append(loss_g.detach().item())
            if idx%3 == 0:
                rand_img = random.randint(0, (len(batch['lr'])-1)//2)
                if batch['lr'][rand_img].dim() > 3:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                else:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                image2 = transforms.ToPILImage()(sr_batch[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text=preds[rand_img], gt_text=batch['gt'][rand_img])

            pbar.set_postfix({'loss': sum(val_loss)/len(val_loss)})
    return sum(val_loss)/len(val_loss), []
 
@register('SROCR_TRAIN')
def SROCR_TRAIN(train_loader, model, optimizer, loss_fn, confusing_pair, *args):
    config = args[0]
    # Set the model to training mode
    model.train()
   # count=0
    train_loss = []
    n_correct=0
    total = 0
    # Zero out the gradients in the optimizer
    optimizer.zero_grad()
    
    # Create a progress bar for training (using tqdm library)
    pbar = tqdm(train_loader, leave=False, desc='train')
    converter = strLabelConverter(config['alphabet'])
    # Iterate through batches in the training data
    for idx, batch in enumerate(pbar):
        # Forward pass: Generate super-resolution (sr) images using the model
        # model = model.cpu()
        preds, sr = model(batch['lr'].cuda())
        preds_all = preds
        # Calculate the loss between the generated sr images and the high-resolution (hr) ground truth images
        
        _, preds_all = preds_all.max(2)
        sim_preds = converter.decode_list(preds_all.data)
        text_label = batch['gt']
        for pred, target in zip(sim_preds, text_label):
            pred = pred.replace('-', '')
            if pred == target:
                n_correct += 1
            total += 1
        
        loss = loss_fn(sr, batch['hr'].cuda(), preds, batch['gt'])        
        
        # Uncomment the following lines if you wish to visualize a random sample.

        if isinstance(sr, tuple):
            Image.fromarray(np.array(transforms.ToPILImage()(sr[0][random.randint(0, len(batch['lr'])-1)].to('cpu'))).astype('uint8')).save(config['model']['name']+'.jpg')
        else:
            if idx%3 == 0:
                rand_img = random.randint(0, len(batch['lr'])-1)
                if batch['lr'][rand_img].dim() > 3:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                else:
                    image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text=sim_preds[rand_img], gt_text=batch['gt'][rand_img])
        
        
        optimizer.zero_grad() # Zero out the gradients in the optimizer (clear previous gradients)
        loss.backward() # Backpropagation: Compute gradients based on the loss
        optimizer.step() # Update the model's parameters using the optimizer
        
        # Append the current loss to the list of training losses
        train_loss.append(loss.detach().item())
        pbar.set_postfix({'loss': sum(train_loss)/len(train_loss)})
        
        loss=None
        sr=None
    #    count+=1
    #    
    #    if count>=10:
    #        break
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))   
    accuracy = (n_correct / float(total))
    print(f'accuracy: {accuracy*100:.2f}%')
    
    # Calculate the average training loss and return it
    return 1-accuracy

# sum(train_loss)/len(train_loss)

@register('SROCR_VAL')
def SROCR_VAL(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    # Set the model to evaluation mode
    model.eval()
    #count=0
    # Create an empty list to store validation losses    
    val_loss = []
    preds_sr_cm = []
    preds_gt_cm = []
    n_correct=0
    total = 0
    # Create a progress bar for validation (using tqdm library)
    pbar = tqdm(val_loader, leave=False, desc='val')
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate through batches in the validation data
        for idx, batch in enumerate(pbar):
            # Forward pass: Generate super-resolution (sr) images using the model
            preds, sr = model(batch['lr'].cuda())
            preds_all = preds
            # Calculate the loss between the generated sr images and the high-resolution (hr) ground truth images
            loss = loss_fn(sr, batch['hr'].cuda(), preds, batch['gt'])        
            _, preds_all = preds_all.max(2)
            sim_preds = converter.decode_list(preds_all.data)
            text_label = batch['gt']
            for pred, target in zip(sim_preds, text_label):
                pred = pred.replace('-', '')
                if pred == target:
                    n_correct += 1
                total += 1 
            print(total)   
            
            if isinstance(sr, tuple):
                Image.fromarray(np.array(transforms.ToPILImage()(sr[0][random.randint(0, len(batch['lr'])-1)].to('cpu'))).astype('uint8')).save(config['model']['name']+'.jpg')
            else:
                if idx%2 == 0:
                    #_, preds_all = preds_all.max(2)
                    #sim_preds = converter.decode_list(preds_all.data)
                    #text_label = batch['gt']
                    #for pred, target in zip(sim_preds, text_label):
                    #    pred = pred.replace('-', '')
                    #    if pred == target:
                    #        n_correct += 1
                    #    total += 1
                    
                    rand_img = random.randint(0, len(batch['lr'])-1) 
                    if batch['lr'][rand_img].dim() > 3:
                         image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                    else:
                         image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                    image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                    image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                    save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png', sr_text=sim_preds[rand_img], gt_text=batch['gt'][rand_img])
            
            # Append the current loss to the list of validation losses
            val_loss.append(loss.detach().item())
            pbar.set_postfix({'loss': sum(val_loss)/len(val_loss)})
            # Update the progress bar to display the running loss
            # pbar.set_postfix({'Running loss': sum(val_loss)/len(val_loss)})
            
            # Clear variables to save memory (optional)
            loss=None
            sr=None
            #count+=1
            #if count>=10:
            #    break
            
    for raw_pred, pred, gt in zip(preds_all, sim_preds, text_label):
        raw_pred = raw_pred.data
        pred = pred.replace('-', '')
        print('raw_pred: %-20s, pred: %-8s, gt: %-8s, match: %s' % (raw_pred, pred, gt, pred==gt))   
    accuracy = (n_correct / float(total))
    print(f'accuracy: {accuracy*100:.2f}%')
    
    if config['CM']:
        flattened_preds = np.concatenate(preds_sr_cm)
        flattened_gts = np.concatenate(preds_gt_cm)
        
        conf_matrix = confusion_matrix(flattened_preds, flattened_gts, labels=range(len(alphabet)))
        conf_matrix_normalized = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        confusing_pairs = extract_confusing_pairs(conf_matrix_normalized, alphabet, 0.9)
    else:
        confusing_pairs = []
    # Calculate the average validation loss and return it
    return 1-accuracy, confusing_pairs

# sum(val_loss)/len(val_loss), confusing_pairs

@register('SR_VAL')
def validation_sr(val_loader, model, loss_fn, confusing_pair, *args):
    config = args[0]
    # Set the model to evaluation mode
    model.eval()
    # count=0
    # Create an empty list to store validation losses    
    val_loss = []
    preds_sr_cm = []
    preds_gt_cm = []
    # Create a progress bar for validation (using tqdm library)
    pbar = tqdm(val_loader, leave=False, desc='val')
    alphabet = config['alphabet']
    converter = strLabelConverter(alphabet)
    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate through batches in the validation data
        for idx, batch in enumerate(pbar):
            # Forward pass: Generate super-resolution (sr) images using the model
            sr = model(batch['lr'].cuda())
            
            # Calculate the loss between the generated sr images and the high-resolution (hr) ground truth images
            if config['CM']:
                loss = loss_fn(sr, batch['hr'].cuda(), None, batch['gt'], confusing_pair)    
                preds_sr_cm.extend(converter.encode_list(loss_fn.OCR_pred(sr)))
                preds_gt_cm.extend(converter.encode_list(batch['gt']).detach().cpu().numpy())
            else:
                loss = loss_fn(sr, batch['hr'].cuda())    
            
            if isinstance(sr, tuple):
                Image.fromarray(np.array(transforms.ToPILImage()(sr[0][random.randint(0, len(batch['lr'])-1)].to('cpu'))).astype('uint8')).save(config['model']['name']+'.jpg')
            else:
                if idx%2 == 0:
                    rand_img = random.randint(0, len(batch['lr'])-1) 
                    if batch['lr'][rand_img].dim() > 3:
                         image1 = transforms.ToPILImage()(batch['lr'][rand_img][rand_img][0:3].to('cpu'))
                    else:
                         image1 = transforms.ToPILImage()(batch['lr'][rand_img][0:3].to('cpu'))
                    image2 = transforms.ToPILImage()(sr[rand_img].detach().to('cpu'))
                    image3 = transforms.ToPILImage()(batch['hr'][rand_img].to('cpu'))
                    save_visualized_images(image1, image2, image3, config['model']['name']+config['tag_view']+'.png')
            
            # Append the current loss to the list of validation losses
            val_loss.append(loss.detach().item())
            pbar.set_postfix({'loss': sum(val_loss)/len(val_loss)})
            # Update the progress bar to display the running loss
            # pbar.set_postfix({'Running loss': sum(val_loss)/len(val_loss)})
            
            # Clear variables to save memory (optional)
            loss=None
            sr=None
            # count+=1
            # if count>=10:
            #     break
    if config['CM']:
        flattened_preds = np.concatenate(preds_sr_cm)
        flattened_gts = np.concatenate(preds_gt_cm)
        
        conf_matrix = confusion_matrix(flattened_preds, flattened_gts, labels=range(len(alphabet)))
        conf_matrix_normalized = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        confusing_pairs = extract_confusing_pairs(conf_matrix_normalized, alphabet, 0.9)
    else:
        confusing_pairs = []
    # Calculate the average validation loss and return it
    return sum(val_loss)/len(val_loss), confusing_pairs
