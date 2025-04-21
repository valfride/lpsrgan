import random
import os
import csv
import cv2
import yaml
import torch
import models
import datasets
import argparse
import numpy as np
import torch.nn as nn
import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from Levenshtein import distance
from train import make_dataloader
from matplotlib import pyplot as plt
import torchvision.transforms as T
import kornia as K
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def resize_fn(img, size):
    return T.ToTensor()(
        T.Resize(size, T.InterpolationMode.BICUBIC)(T.ToPILImage()(img))
    )

def prepare_testing():
    # Create a data loader for the test dataset
    test_loader = make_dataloader(config['test_dataset'], tag='test')
    
    # Load the super-resolution (SR) model
    sv_file = config['model']
    sv_file = torch.load(sv_file['load'])
    
    # Create the SR model based on the loaded model specifications
    model_sr = models.make(sv_file['model'], load_model=True).cuda()
    
    # Check the number of available GPUs
    n_gpus = torch.cuda.device_count()
    
    # If multiple GPUs are available, use DataParallel to parallelize the SR model
    # if n_gpus > 1:
    #     model_sr = nn.parallel.DataParallel(model_sr)
    
    # Load the OCR model based on the configuration
    if config['model_ocr']['name'] == 'ocr':
        model_ocr = models.make(config['model_ocr'])
    else:
        sv_file = config['model_ocr']
        sv_file = torch.load(sv_file['load'])
        model_ocr = models.make(sv_file['model'], load_model=True).cuda()
    
    # Set the Torch RNG (Random Number Generator) state based on the loaded state
    state = sv_file['state']
    torch.set_rng_state(state)
    
    # Return the test data loader, the SR model, and the OCR model
    return test_loader, model_sr, model_ocr

def build_character_accuracy_histogram(ground_truth, predictions, bar_width=0.6, space_between_bars=1.5, figure_size=(20, 12), title_postfix = 'brazilian'):
    """
    Build and plot a stylized histogram showing the percentage of correct predictions
    for each character, relative to the total number of occurrences of that character,
    and display the total number of occurrences in the labels.
    
    Parameters:
    ground_truth (list of str): The list of correct license plate strings.
    predictions (list of str): The list of license plate strings predicted by the OCR.
    bar_width (float): The width of each bar in the histogram.
    space_between_bars (float): The space between bars in the histogram.
    figure_size (tuple): The size of the figure (width, height).
    
    Returns:
    char_accuracy_percentage (dict): A dictionary with characters as keys and their correct prediction percentage as values.
    """
    
    total_characters_count = Counter()
    correct_characters_count = Counter()

    for pred, gt in zip(predictions, ground_truth):
        for p_char, g_char in zip(pred, gt):
            total_characters_count[g_char] += 1  # Count every occurrence of the ground truth character
            if p_char == g_char:
                correct_characters_count[g_char] += 1  # Count correct predictions

    characters = sorted(total_characters_count.keys())
    char_accuracy_percentage = {char: (correct_characters_count[char] / total_characters_count[char]) * 100 
                                for char in characters}

    # Calculate bar positions with extra space
    positions = np.arange(len(characters)) * (bar_width + space_between_bars)

    # Set the figure size
    plt.figure(figsize=figure_size)

    counts = [char_accuracy_percentage[char] for char in characters]
    plt.bar(positions, counts, width=bar_width, color='lightgreen')
    plt.xlabel('Character')
    plt.ylabel('Correct Prediction Percentage (%)')
    plt.title(f'OCR Character Prediction Accuracy Percentage - {title_postfix}' )

    # Add percentages above the bars
    for i, percentage in enumerate(counts):
        plt.text(positions[i], percentage + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')

    # Set custom x-ticks with corresponding labels
    character_labels = [f'{char}\n{total_characters_count[char]}' for char in characters]
    plt.xticks(positions, character_labels)

    plt.show()
    
    return char_accuracy_percentage

def build_ocr_accuracy_histogram(ground_truth, predictions, title_postfix = 'brazilian'):
    """
    Build and plot a stylized histogram showing the number of license plates 
    with a specific number of correctly predicted characters, including totals.
    
    Parameters:
    ground_truth (list of str): The list of correct license plate strings.
    predictions (list of str): The list of license plate strings predicted by the OCR.
    
    Returns:
    histogram (Counter): A counter object representing the frequency of correct character counts.
    """
    
    def count_correct_characters(pred, gt):
        return sum(p == g for p, g in zip(pred, gt))

    histogram = Counter()

    for pred, gt in zip(predictions, ground_truth):
        correct_count = count_correct_characters(pred, gt)
        histogram[correct_count] += 1

    total_lps = len(predictions)
    histogram_list = [histogram[i] for i in range(8)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(8), histogram_list, tick_label=range(8), color='skyblue')
    plt.xlabel('Number of Correct Characters')
    plt.ylabel('Number of License Plates')
    plt.title(f'OCR Prediction Accuracy Histogram - {title_postfix} (Total LPs: {total_lps})')

    for i, count in enumerate(histogram_list):
        plt.text(i, count + 0.2, str(round((count/total_lps)*100, 1))+'%', ha='center', va='bottom')

    plt.show()
    
    return histogram

def test(test_loader, model_sr, model_ocr, save_path):
    # Set the SR model to evaluation mode
    model_sr.eval()
    counter = 0
    # Create a progress bar for visualizing the testing progress
    pbar = tqdm(test_loader, leave=False, desc='test')
    
    # Initialize a list to store predictions
    preds = []
    
    # Create a directory for saving the test result images
    results_path = save_path / Path('imgs')
    results_path.mkdir(parents=True, exist_ok=True)
    ground_truth = []
    predictions = []
    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            # If the input LR image is a list (possibly multiple LR images), move them to GPU
            if isinstance(batch['lr'], list):
                batch['lr'][0], batch['lr'][1] = batch['lr'][0].cuda(), batch['lr'][1].cuda()                
                
            # Generate super-resolved (SR) images using the SR model
            _ ,sr = model_sr(batch['lr'].cuda())         
            
            # If the SR output is a tuple, extract the relevant part (assuming it's the first element)
            if isinstance(sr, tuple):
                sr = sr[0].cuda()
                
            # Process each SR image, extract OCR predictions, and save images
            for img, lr, hr, lp, name in zip(sr, batch['lr'], batch['hr'], batch['gt'], batch['name']):
                # Convert the SR image from PyTorch tensor to NumPy array and adjust color channels
                # img = K.enhance.equalize_clahe(img, clip_limit=4.0, grid_size=(2, 2))
                if config['model_ocr']['name'] == 'ocr':
                    img = img.cpu().numpy().transpose(1, 2, 0)          
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # lr = T.ToPILImage()(lr[3:6, :, :])
                if lr.dim() > 3:
                    lr = T.ToPILImage()(lr[0].to('cpu'))
                elif lr.size(2) > 3:
                    lr = T.ToPILImage()(lr[0:3])
                else:
                    lr = T.ToPILImage()(lr)
                hr = T.ToPILImage()(hr)
                save_dest = results_path / '/'.join(str(name).split('/')[-4:])
                save_dest = save_dest.parent
                save_dest.mkdir(parents=True, exist_ok=True)
                # print(Path(Path("lr_") + name))
                lr.save(save_dest / Path("lr_" + name.name).with_suffix('.png'))
                hr.save(save_dest / Path("hr_" + name.name).with_suffix('.png'))
                
                # Use the OCR model to predict text from the SR image
                pred = model_ocr.OCR_pred(img.unsqueeze(0))[0][0].replace('#', '')
                if config['model_ocr']['name'] == 'ocr':
                    lr = cv2.cvtColor((np.asarray(lr)/255.0).astype('float32'), cv2.COLOR_RGB2BGR)
                    hr = cv2.cvtColor((np.asarray(hr)/255.0).astype('float32'), cv2.COLOR_RGB2BGR)
                    predLR = model_ocr.OCR_pred(lr)[0].replace('#', '')
                    predHR = model_ocr.OCR_pred(hr)[0].replace('#', '')
                else:
                    hr = T.ToTensor()(hr)
                    lr = T.ToTensor()(lr)
                    lr = resize_fn(lr, (hr.size(1), hr.size(2)))
                    predLR = model_ocr.OCR_pred(lr.unsqueeze(0).cuda())[0][0].replace('#', '')
                    predHR = model_ocr.OCR_pred(hr.unsqueeze(0).cuda())[0][0].replace('#', '')
                

                #.replace('#', '')
                # Calculate accuracy by measuring the edit distance between predicted and ground truth text
                # Append the prediction details to the 'preds' list
                preds.append({'PredSR': pred, 'PredLR': predLR, 'PredHR': predHR, 'Gt': lp, 'AccLR': len(lp) - distance(predLR, lp), 'AccHR': len(lp) - distance(predHR, lp), 'Acc': len(lp) - distance(pred, lp), 'Name': name})
                print(f"predSR: {pred}, predHR: {predHR}, predLR: {predLR}")
                # Save the SR image
                if config['model_ocr']['name'] == 'ocr':
                    img = Image.fromarray((cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*255).astype(np.uint8))
                else:
                    img = T.ToPILImage()(img)
                img.save(save_dest / Path("sr_" + name.name).with_suffix('.png'))
                
            # counter+=1
            # print(counter)
            # if counter >= 10:
            #     break
            # break
        # histogram = build_ocr_accuracy_histogram(ground_truth, predictions)
        # char_histogram = build_character_accuracy_histogram(ground_truth, predictions)
        # print(histogram)
        # print(char_histogram)
            
        # Save the test results to a CSV file
        with open(save_path / Path('results.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = ['PredSR', 'PredLR', 'PredHR', 'Gt', 'AccLR', 'AccHR', 'Acc', 'Name'])
            writer.writeheader()
            writer.writerows(preds)

def main(config_, save_path):
    global config
    config = config_    
         
    # Call the prepare_testing function to set up testing
    test_loader, model_sr, model_ocr = prepare_testing()

    # Call the test function to perform the testing
    test(test_loader, model_sr, model_ocr, save_path)
    

if __name__ == '__main__':            
    # Create an argument parser to parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)    
    parser.add_argument('--tag', default=None)

    # Parse the command line arguments
    args = parser.parse_args()
    # Define a function to set random seeds for reproducibility
    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # sets the seed for cpu
        torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
        torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Set a fixed random seed (for reproducibility)
    setup_seed(1996)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Create a save_name based on the configuration file and tag
    save_name = args.save
    if save_name is not None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    
    # Create a save_path directory for saving the test results
    save_path = Path('./multi_image_research/test') / Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)    

    # Call the main function to start the testing process
    main(config, save_path)
