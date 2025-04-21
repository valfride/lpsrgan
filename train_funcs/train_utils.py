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
