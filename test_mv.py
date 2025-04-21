import os
import csv
import cv2
import yaml
import torch
import models
import copy
import pandas as pd
import datasets
import argparse
import numpy as np
import torch.nn as nn
import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import Levenshtein
from train import make_dataloader
from collections import defaultdict

from matplotlib import pyplot as plt
from utils_test import majority_vote_by_character, select_highest_confidence_string, select_most_frequent_string
import torchvision.transforms as T
import kornia as K
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
import torch.nn.functional as F
import random
import time
from pynvml import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def resize_fn(img, size):
    return T.ToTensor()(
        T.Resize(size, T.InterpolationMode.BICUBIC)(T.ToPILImage()(img))
    )

def prepare_testing():
    # Create a data loader for the test dataset
    test_loader = make_dataloader(config['test_dataset'], tag='test')
    
    if config['model'] is not None:
        # Load the super-resolution (SR) model
        sv_file = config['model']
        sv_file = torch.load(sv_file['load'])
        # print(len(sv_file))
        # Create the SR model based on the loaded model specifications
        model_sr = models.make(sv_file['model'], load_model=True).cuda()
    else:
        model_sr = None
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

def process_results(preds, preds_conf_mean, t, mv_character_results, mv_hc_results, mv_results, res_type):
    # Append results to each dictionary for the specified resolution type and threshold
    mv_character_results[res_type][t].append(majority_vote_by_character(preds, t))
    mv_hc_results[res_type][t].append(select_highest_confidence_string(preds_conf_mean, preds, t))
    mv_results[res_type][t].append(select_most_frequent_string(preds_conf_mean, preds, t))

def calculate_levenshtein_batch(predictions, gt_batch):
    lev_distances = {}

    # Loop over each image in the batch
    for idx, gt in enumerate(gt_batch):
        lev_distances[idx] = {}  # Create a dictionary for each image

        # Loop over resolution types ('lr', 'hr', 'sr')
        for res_type, thresholds in predictions.items():
            lev_distances[idx][res_type] = {}

            # Loop over thresholds (1, 3, 5)
            for t, pred_list in thresholds.items():
                # Get the predicted strings for the current threshold
                pred_str = pred_list[idx]  # Use the prediction for the current image in the batch
                gt_str = gt  # Ground truth string (assuming it's a list with a single string)
                # Calculate the Levenshtein distance between predicted and ground truth
                lev_dist = Levenshtein.distance(pred_str, gt_str)
                # print(len(pred_str) > 7)
                lev_distances[idx][res_type][t] = 7-lev_dist

    return lev_distances

def count_distances_and_percentages_by_res_and_threshold(levenshtein_results, total_images, upper=1):
    # Initialize dictionaries to store the counts of distances for each resolution type and threshold
    distance_counts_by_res_and_threshold = {
        'lr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'hr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)},
        'sr': {1: defaultdict(int), 3: defaultdict(int), 5: defaultdict(int)}
    }
    
    # Initialize dictionaries to store the percentages of distances for each resolution type and threshold
    distance_percentages_by_res_and_threshold = {
        'lr': {1: {}, 3: {}, 5: {}},
        'hr': {1: {}, 3: {}, 5: {}},
        'sr': {1: {}, 3: {}, 5: {}}
    }

    # Iterate through each image's Levenshtein results
    for idx, res_data in levenshtein_results.items():
        # Iterate through each resolution type ('lr', 'hr', 'sr')
        for res_type, thresholds in res_data.items():
            # Iterate through each threshold (1, 3, 5)
            for t, dist in thresholds.items():
                # Increment the count for the corresponding distance for the specific res_type and threshold
                distance_counts_by_res_and_threshold[res_type][t][dist] += 1

    # Calculate the percentages for each resolution type and threshold
    for res_type in ['lr', 'hr', 'sr']:
        for t in [1, 3, 5]:
            for dist, count in distance_counts_by_res_and_threshold[res_type][t].items():
                distance_percentages_by_res_and_threshold[res_type][t][dist] = round((count / total_images) * 100, upper)

    return distance_counts_by_res_and_threshold, distance_percentages_by_res_and_threshold

def calculate_distances_and_percentages(mv_character_results, mv_hc_results, mv_results, gt):
    """Calculate Levenshtein results and distance percentages."""
    levenshtein_results_mvcp = calculate_levenshtein_batch(mv_character_results, gt)
    levenshtein_results_hc = calculate_levenshtein_batch(mv_hc_results, gt)
    levenshtein_results_mv = calculate_levenshtein_batch(mv_results, gt)

    _, distance_percentages_mvcp = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_mvcp, len(levenshtein_results_mvcp))
    _, distance_percentages_hc = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_hc, len(levenshtein_results_hc))
    _, distance_percentages_mv = count_distances_and_percentages_by_res_and_threshold(levenshtein_results_mv, len(levenshtein_results_mv))
    
    return distance_percentages_mvcp, distance_percentages_hc, distance_percentages_mv, levenshtein_results_mvcp, levenshtein_results_hc, levenshtein_results_mv


def create_threshold_table(data, thresholds=[7, 6, 5]):
    tables = {}
    for res_type, threshold_data in data.items():
        df = pd.DataFrame(threshold_data).transpose()
        df_cumulative = pd.DataFrame({f'>={t}': df[df.columns[df.columns >= t]].sum(axis=1) for t in thresholds})
        tables[res_type] = df_cumulative
    return tables

def write_mv_character_results_to_csv(table, mv_character_results, mv_dist, mv_path_images, gt, save_path, name):
    """
    Writes the contents of mv_character_results, distances, paths, and ground truths to a CSV file.
    Also saves a separate summary CSV file with table data.

    Parameters:
        table (dict): Dictionary containing summary tables (e.g., LR, HR, SR) as pandas DataFrames.
        mv_character_results (dict): Dictionary with main character recognition results.
        mv_dist (dict): Dictionary with Levenshtein distances or similar metrics.
        mv_path_images (list): List of paths or identifiers for each image row.
        gt (list): List of ground truth values for each row.
        save_path (Path or str): Directory path to save the output CSV file.
        name (str): Name of the main output CSV file.
    """
    # Define paths for the main and summary CSV files
    output_path = Path(save_path) / name
    summary_output_path = Path(save_path) / f"summary_{name}"

    # Get categories and indices from the mv_character_results structure
    categories = list(mv_character_results.keys())
    indices = list(mv_character_results[categories[0]].keys())

    # Construct the CSV header
    header = ["path", "Gt"] + [f"{cat}_{idx}" for cat in categories for idx in indices] + \
             [f"#correct_{cat}_{idx}" for cat in categories for idx in indices]

    # Write the main CSV file
    with output_path.open(mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        
        # Determine the number of rows based on the length of lists in mv_character_results
        num_rows = len(next(iter(mv_character_results['lr'].values())))  # Assumes uniform length across all lists
        
        # Write each row to the CSV
        for row_idx in range(num_rows):
            # Construct the row with a single list comprehension for efficient data gathering
            row = [
                str(mv_path_images[row_idx]), str(gt[row_idx]),  # Convert path and gt to strings
                *[mv_character_results[cat][idx][row_idx] for cat in categories for idx in indices],
                *[mv_dist[row_idx][cat][idx] for cat in categories for idx in indices]
            ]
            writer.writerow(row)

    # Write summary tables to a separate CSV file
    with summary_output_path.open(mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        
        # Iterate through each category in the table and write the data
        for cat, data in table.items():
            # Write category section header
            writer.writerow([cat])
            
            # Round data to 1 decimal place
            data = data.round(1)
            
            # Write DataFrame headers and contents to CSV
            writer.writerow([''] + list(data.columns))  # Column headers with an empty leading cell for row indices
            writer.writerows([[idx] + row.tolist() for idx, row in data.iterrows()])  # Each row with index

def test(test_loader, model_sr, model_ocr, save_path):
    # Set the SR model to evaluation mode
    if model_sr is not None:
        model_sr.eval()
    model_ocr.eval()
    # Create a progress bar for visualizing the testing progress
    pbar = tqdm(test_loader, leave=False, desc='test')
    count = 0
    # Initialize a list to store predictions
    preds = []
    
    # Create a directory for saving the test result images
    results_path = save_path / Path('imgs')
    results_path.mkdir(parents=True, exist_ok=True)
    
    gt = []
    
    total_time = 0
    timings = []
    
    with torch.no_grad():
    # Initialize result storage outside the batch loop to accumulate across batches
        results_dict_template = {1: [], 3: [], 5: []}
        mv_character_results = {k: defaultdict(list, copy.deepcopy(results_dict_template)) for k in ['lr', 'hr', 'sr']}
        mv_hc_results = {k: defaultdict(list, copy.deepcopy(results_dict_template)) for k in ['lr', 'hr', 'sr']}
        mv_results = {k: defaultdict(list, copy.deepcopy(results_dict_template)) for k in ['lr', 'hr', 'sr']}
        mv_path_images = []
        # Loop through batches
        for idx, batch in enumerate(pbar):
            # Prepare images and move them to GPU
            imgs_hr = batch['hr'].view(-1, 3, batch['hr'].size(2), batch['hr'].size(3)).cuda()
            if model_sr is not None:
                imgs_lr = batch['lr'].view(-1, 3, 16, 48)
                start = time.time()
                imgs_sr = model_sr(imgs_lr.cuda())
                timings.append((time.time() - start)*1000)  # Convert to ms
                # print('LR:', imgs_lr.shape, 'HR', imgs_hr.shape, 'SR', imgs_sr.shape)
            else:
                imgs_sr = batch['lr'].view(-1, 3, 32, 96)
                imgs_lr = batch['lr'].view(-1, 3, 32, 96)
                imgs_sr = F.interpolate(imgs_sr, size=(imgs_hr.size(2), imgs_hr.size(3)), mode='bilinear', align_corners=False).cuda()
            
            # Perform OCR predictions for each resolution
            preds_dict = {
                'lr': model_ocr.OCR_pred(F.interpolate(imgs_lr, size=(imgs_hr.size(2), imgs_hr.size(3)), mode='bilinear', align_corners=False).cuda()),
                'hr': model_ocr.OCR_pred(imgs_hr),
                'sr': model_ocr.OCR_pred(imgs_sr)
            }
            # print(preds_dict)
            gt.append(batch['gt'][0])
            # print(f"GT: {gt[0]}  PRED: {preds_dict['sr'][0]}  Path: {batch['name']}")
            mv_path_images.append(batch['name'])
            # For each resolution and threshold, calculate and store results
            
            
            
            for res_type, (preds, preds_conf_mean) in preds_dict.items():
            
                for t in [1, 3, 5]:
                    process_results(preds, preds_conf_mean, t, mv_character_results, mv_hc_results, mv_results, res_type)
            count+=1
            if count > 100:
                sample_sizes = [5, 10, 25, 50, 100]
                results = {}
                
                for size in sample_sizes:
                    # Ensure we don't exceed available data
                    current_sample = timings[1:min(size+1, len(timings))]
                    
                    results[size] = {
                        'avg': np.mean(current_sample),
                        'std': np.std(current_sample),
                        'min': np.min(current_sample),
                        'max': np.max(current_sample),
                        'n': len(current_sample)
                    }
                
                # Print formatted results
                print(f"{'Sample':<8} {'Avg':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'N':<5}")
                print("-" * 60)
                for size, data in results.items():
                    print(f"{size:<8} {data['avg']:.6f} ± {data['std']:.6f} {data['min']:.6f} {data['max']:.6f} {data['n']:<5}")
                
                # Alternative: Dictionary output for further processing
                print("\nComplete results:")
                print(results)
                
                break
            
            img_save_path = results_path / batch['name'][0].parent
            img_save_path.mkdir(parents=True, exist_ok=True)
            for i, (img_lr, img_sr) in enumerate(zip(imgs_lr, imgs_sr)):
                filename_lr = f"lr-{i+1:03}.png"
                filename_sr = f"sr-{i+1:03}.png"
                
                img_lr = T.ToPILImage()(img_lr)
                img_sr = T.ToPILImage()(img_sr)
                img_lr.save(img_save_path / Path(filename_lr))
                img_sr.save(img_save_path / Path(filename_sr))
        
        distance_percentages_mvcp, distance_percentages_hc, distance_percentages_mv, distance_mvcp, distance_hc, distance_mv = calculate_distances_and_percentages(mv_character_results, mv_hc_results, mv_results, gt)

        # Create a DataFrame with the percentage results for each distance type
        hc_table = create_threshold_table(distance_percentages_hc)
        mvcp_table = create_threshold_table(distance_percentages_mvcp)
        mv_table = create_threshold_table(distance_percentages_mv)
        
        write_mv_character_results_to_csv(mvcp_table, mv_character_results, distance_mvcp,  mv_path_images, gt, save_path, "MVCP.csv")
        write_mv_character_results_to_csv(hc_table, mv_hc_results, distance_hc, mv_path_images, gt, save_path, "HC.csv")
        write_mv_character_results_to_csv(mv_table, mv_results, distance_mv, mv_path_images, gt, save_path, "MV.csv")
        
        
        # print(total_time/100)
        # for res_type, df in mvcp_table.items():
        #     print(f"{res_type.upper()} Table:")
        #     print(df)
        #     print("\n")
            
        # for res_type, df in hc_table.items():
        #     print(f"{res_type.upper()} Table:")
        #     print(df)
        #     print("\n")
            
        # for res_type, df in mv_table.items():
        #     print(f"{res_type.upper()} Table:")
        #     print(df)
        #     print("\n")
 
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
    
    def reset_gpu():
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Reset NVIDIA driver cache (requires nvidia-ml-py)
        try:
            nvmlInit()
            for i in range(torch.cuda.device_count()):
                handle = nvmlDeviceGetHandleByIndex(i)
                nvmlDeviceResetGpuLockedClocks(handle)
            nvmlShutdown()
        except:
            pass  # Fallback if pynvml not installed
    
    # Usage
    reset_gpu()
    
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
    save_name = Path(args.save)
    if save_name is not None:
        save_name = save_name / Path(args.config.split('/')[-1][:-len('.yaml')])
    if args.tag is not None:
        save_name = Path(str(save_name) + '_' + args.tag)
    
    # Create a save_path directory for saving the test results
    save_path = Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)

    # Call the main function to start the testing process
    main(config, save_path)
