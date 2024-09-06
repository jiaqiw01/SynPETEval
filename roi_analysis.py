import nibabel as nib
import numpy as np
import os
import glob
import h5py
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns
import time
from scipy.stats import chisquare, wasserstein_distance
sns.set_context("paper")
sns.set_style("darkgrid")
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import scipy.ndimage as ndimage
import skimage
import piq
import utils
import argparse

parser = argparse.ArgumentParser(prog='Roi Subject Analysis')
parser.add_argument('-d', '--dir', nargs='+', default=["/home/jiaqiw01/test_cases_fdg_1p_conditioned"])
parser.add_argument('-e', '--exp_types', nargs='+', default=["GAN_t1_t2f_pet_1p"])
parser.add_argument('-r', '--result', default="/home/jiaqiw01/test_roi_analysis")
parser.add_argument('-l', '--label', default="")


###### Histogram related #####

def get_histogram(data, num_bins=64, scale=10, output='norm'):
     # calculate histogram
    histograms, _ = np.histogram(data, bins=num_bins, range=(0.001, 1))
    normalized_histograms = histograms / (histograms.sum(keepdims=True) + 1e-4)
    normalized_histograms *= scale
    if output == 'norm':
        return normalized_histograms
    cum_hist = np.cumsum(normalized_histograms)
    hist_diff = np.diff(normalized_histograms)
    hist_diff = np.insert(hist_diff, 0, hist_diff[0])
    hist_diff *= scale
    combined_histogram = np.stack((normalized_histograms, cum_hist, hist_diff), axis=0) # 3, 128
    # assert combined_histogram.shape == (3, 128)
    return combined_histogram
    
def compare_histograms(vol1, vol2, roi_name, pet_type, hist_truth_left, hist_truth_right, emd_truth, num_bins=128):
    '''
        Evaluate histogram with Wasserstein loss, can be a single slice or a whole volume
    '''
    hist1 = get_histogram(vol1, num_bins, output='all')[0]
    hist2 = get_histogram(vol2, num_bins, output='all')[0]
    emd = wasserstein_distance(hist1, hist2)
    plt.clf()
    plt.subplot(141)
    plt.bar(range(num_bins), hist1, width=3)
    plt.subplot(142)
    plt.bar(range(num_bins), hist2, width=3)
    plt.subplot(143)
    plt.bar(range(num_bins), hist_truth_left, width=3)
    plt.subplot(144)
    plt.bar(range(num_bins), hist_truth_right, width=3)
    plt.title(f"{pet_type} {roi_name} Asymmetry Wasserstein {emd} VS {emd_truth}")
    plt.show()
    # Wasserstein distance
    print(f'{pet_type} {roi_name} EMD: {emd}')
    return emd

def compare_histogram_asymmetry(roi_masks, pets, num_bins=128):
    res = {}
    for roi, val in roi_masks.items():
        roi_mask_left = val['left']
        roi_mask_right = val['right']
        if roi not in res:
            res[roi] = {}
        hist_truth_left = get_histogram(pets['truth'] * roi_mask_left, num_bins=num_bins)
        hist_truth_right = get_histogram(pets['truth'] * roi_mask_right, num_bins=num_bins)
        emd_truth = wasserstein_distance(hist_truth_left, hist_truth_right)
        for pet_type, pet_vol in pets.items():
            if pet_type == 'truth':
                continue
            vol_left = pet_vol * roi_mask_left
            vol_right = pet_vol * roi_mask_right
            distance = compare_histograms(vol_left, vol_right, roi, pet_type, hist_truth_left, hist_truth_right, emd_truth)
            res[roi][pet_type] = distance
    return res


###### SUVR Related #####
def compute_suvr(roi_masks, ref_name, pet, include_diff=True): # {pet1: {roi1: {'suvr': xx, ...}}, pet2: {roi1: {'suvr': xx}}}
    ref_mask = roi_masks[ref_name]
    if type(pet) == dict:
        res = {}
        res_diff = {}
        pet_data = pet['truth']
        res['truth'] = {}
        suv_ref = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        for roi in roi_masks.keys():
            curr_roi_mask = roi_masks[roi]
            suv_roi = (pet_data * curr_roi_mask).sum() / curr_roi_mask.sum() # roi total suv / roi total volume
            suvr_roi = suv_roi / (suv_ref + 1e-5)
            res['truth'][roi] = {'suvr': suvr_roi}
        for p in pet.keys():
            if p == 'truth':
                continue
            res[p] = {}
            pet_data = pet[p]
            suv_ref = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
            for roi in roi_masks.keys():
                curr_roi_mask = roi_masks[roi]
                suv_roi = (pet_data * curr_roi_mask).sum() / curr_roi_mask.sum() # roi total suv / roi total volume
                suvr_roi = suv_roi / (suv_ref + 1e-5)
                res[p][roi] = {'suvr': suvr_roi}
                if include_diff:
                    if p not in res_diff:
                        res_diff[p] = {}
                    res_diff[p][roi] = res['truth'][roi]['suvr'] - suvr_roi # how suvr deviates from the true pet
        return res, res_diff
    else:
        raise NotImplementedError

def compute_suvr_asymmetry(hemisphere_roi_masks, global_roi_masks, pet, include_diff=True, ref_name='cerebellum'):
    ref_mask = global_roi_masks[ref_name]
    res = {}
    pet_data = pet['truth']
    res['truth'] = {}
    res_diff = {}
    suv_ref_global = (pet_data * ref_mask).sum() / ref_mask.sum() # global reference, i.e., both left & right cerebellum
    for roi in hemisphere_roi_masks.keys():
        curr_roi_mask_left = hemisphere_roi_masks[roi]['left']
        curr_roi_mask_right = hemisphere_roi_masks[roi]['right']
        suv_roi_left = (pet_data * curr_roi_mask_left).sum() / curr_roi_mask_left.sum() # roi total suv / roi total volume
        suvr_roi_left = suv_roi_left / (suv_ref_global + 1e-4)
        
        suv_roi_right = (pet_data * curr_roi_mask_right).sum() / curr_roi_mask_right.sum() # roi total suv / roi total volume
        suvr_roi_right = suv_roi_right / (suv_ref_global + 1e-4)
        
        res['truth'][roi] = {'left': suvr_roi_left,
                            'right': suvr_roi_right,
                            'diff': (suvr_roi_left - suvr_roi_right) / (suvr_roi_left + suvr_roi_right)}
    for p in pet.keys():
        if p == 'truth':
            continue
        res[p] = {}
        pet_data = pet[p]
        suv_ref_global = (pet_data * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        for roi in hemisphere_roi_masks.keys():
            curr_roi_mask_left = hemisphere_roi_masks[roi]['left']
            curr_roi_mask_right = hemisphere_roi_masks[roi]['right']
            suv_roi_left = (pet_data * curr_roi_mask_left).sum() / curr_roi_mask_left.sum() # roi total suv / roi total volume
            suvr_roi_left = suv_roi_left / (suv_ref_global + 1e-4)
            
            suv_roi_right = (pet_data * curr_roi_mask_right).sum() / curr_roi_mask_right.sum() # roi total suv / roi total volume
            suvr_roi_right = suv_roi_right / (suv_ref_global + 1e-4)
            res[p][roi] = {'left': suvr_roi_left,
                           'right': suvr_roi_right,
                           'diff': (suvr_roi_left - suvr_roi_right) / (suvr_roi_left + suvr_roi_right)}
            if include_diff:
                if p not in res_diff:
                    res_diff[p] = {}
                res_diff[p][roi] = {'left': res['truth'][roi]['left'] - res[p][roi]['left'],
                                   'right': res['truth'][roi]['right'] - res[p][roi]['right'],
                                    'diff': res['truth'][roi]['diff'] - res[p][roi]['diff'],
                                   }
    return res, res_diff

##### Some common metrics #####
def compute_metrics(all_pets):
    res = {}
    truth = utils.norm(all_pets['truth'][:, :, 20:71])
    for pet in all_pets.keys():
        if pet == 'truth':
            continue
        curr_pred_img = utils.norm(all_pets[pet][:, :, 20:71])
        psnr = skimage.metrics.peak_signal_noise_ratio(truth, curr_pred_img, data_range=1)
        ssim = skimage.metrics.structural_similarity(truth, curr_pred_img, data_range=1)
        rmse = skimage.metrics.normalized_root_mse(truth, curr_pred_img)

        slice_true_torch = torch.Tensor(truth).unsqueeze(0).permute(3, 0, 1, 2)
        slice_pred_torch = torch.Tensor(curr_pred_img).unsqueeze(0).permute(3, 0, 1, 2)
        msssim = piq.multi_scale_ssim(slice_true_torch, slice_pred_torch, data_range=1.)
        res[pet] = {'psnr': psnr, 'rmse': rmse, 'ssim': ssim, 'msssim': msssim}
        print(f"PET: {pet}, PSNR: {psnr}, MSSSIM: {msssim}, RMSE: {rmse}, SSIM: {ssim}")
    return res

##### Bland altman for all test subjects #####
def global_bland_altman(combined_df, suffix, target_pet_type, true_pet_type, shared_rois, save_name):
    '''
        Bland altman for ROIs, taking a dataframe
        suffix: _suvr or _diff
        _diff is for SUVR asymmetry assessment
        _suvr is for SUVR assessment
    '''
    target_cols = [x+suffix for x in shared_rois]
    pet_pred_df = combined_df[combined_df['pet_type'] == target_pet_type].sort_values(by=['subject'])[target_cols]
    pet_true_df = combined_df[combined_df['pet_type'] == true_pet_type].sort_values(by=['subject'])[target_cols]
    suvr_dict_gt_list = []
    suvr_dict_pred_list = []
    suvr_dict_gt = {}
    suvr_dict_pred = {}
    for roi in shared_rois:
        col_val = pet_pred_df[roi+suffix].values
        suvr_dict_pred[roi] = col_val
        suvr_dict_pred_list.extend(col_val)
        col_val = pet_true_df[roi+suffix].values
        suvr_dict_gt[roi] = col_val
        suvr_dict_gt_list.extend(col_val)
        
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightblue', 'limegreen', 'black', 'maroon']
    plt.figure(figsize=(8,6))
    
    for i, lab in enumerate(suvr_dict_gt.keys()):
        # print(suvr_dict_gt[lab], suvr_dict_pred[lab])
        plt.plot(0.5*(np.array(suvr_dict_gt[lab])+np.array(suvr_dict_pred[lab])), np.array(suvr_dict_gt[lab])-np.array(suvr_dict_pred[lab]), '.', alpha=0.6, color=color_roi_mapping[lab], label=lab, markersize=10)
    
    sd196 = round(1.96*np.std(np.array(suvr_dict_gt_list)-np.array(suvr_dict_pred_list)), 2)
    mean = round(np.mean(np.array(suvr_dict_gt_list)-np.array(suvr_dict_pred_list)), 2)
    xmin = round(0.5*np.min(np.array(suvr_dict_gt_list)+np.array(suvr_dict_pred_list)), 2)
    xmax = round(0.5*np.max(np.array(suvr_dict_gt_list)+np.array(suvr_dict_pred_list)), 2)
    plt.plot([xmin, xmax], [mean+sd196, mean+sd196], color='tab:orange', linestyle='dotted')
    plt.text(xmax-0.05, mean+sd196+0.02, "%.2f" % (mean+sd196), fontsize=10)
    plt.plot([xmin, xmax], [mean-sd196, mean-sd196], color='tab:orange', linestyle='dotted')
    plt.text(xmax-0.05, mean-sd196+0.02, "%.2f" % (mean-sd196), fontsize=10)
    plt.plot([xmin, xmax], [mean, mean], color='tab:purple', linestyle='dotted')
    plt.legend(ncol=2)
    plt.text(xmax-0.05, mean+0.02, "%.2f" % mean, fontsize=10)
    plt.xlabel('Mean of methods', fontsize=14)
    plt.ylabel('Acquired - Synthesized SUVR', fontsize=14)
    # plt.savefig('global_bland_altman.png')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Adjust the plot to make room for the legend
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def prepare_all_subjects(test_dirs, exp_types, save_dest):
    test_subjects = os.listdir(test_dirs[0])
    all_suvr_diff_df = []
    all_hemi_diff_df = []
    all_suvr_df = []
    all_subj_metrics = {}
    all_subj_suvr_diff = {}

    fig1, axes = plt.subplots(nrows=4, ncols=len(test_subjects)//4 + 1, figsize=(15,25))
    axes_flatten_1 = axes.ravel()

    fig2, axes = plt.subplots(nrows=4, ncols=len(test_subjects)//4 + 1, figsize=(15, 25))
    axes_flatten_2 = axes.ravel()
    
    
    for i, subj in enumerate(test_subjects):
        print(f"------Performing {subj}--------")
        suvr, suvr_diff, all_metrics, hemi_diff, hemi_diff_diff =  analyze_one_subj(subj, exp_types, save_dest, *test_dirs)
        utils.plot_asymmetry(hemi_diff, subj, axes_flatten_1[i])

        suvr_df = utils.json_to_pd_row(subj, suvr)
        suvr_diff_df = utils.json_to_pd_row(subj, suvr_diff)
        hemi_diff_df = utils.json_to_pd_row(subj, hemi_diff)
    
        all_suvr_diff_df.append(suvr_diff_df)
        all_suvr_df.append(suvr_df)
        all_hemi_diff_df.append(hemi_diff_df)

        all_subj_metrics[subj] = all_metrics
        all_subj_suvr_diff[subj] = suvr_diff
        utils.plot_suvr_diff_one_subj(suvr_diff, subj, axes_flatten_2[i], save_name=None)
        print("------Done------")

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flatten_1)):
        fig1.delaxes(axes_flatten_1[j])
        fig2.delaxes(axes_flatten_2[j])
        
    plt.tight_layout()
    if save_dest:
        fig1.savefig(os.path.join(save_dest, "SUVR_Asymmetry.png"), bbox_inches='tight', dpi=300)
        fig1.savefig(os.path.join(save_dest, "SUVR_Differences.png"), bbox_inches='tight', dpi=300)
    else:
        fig1.show()
        fig2.show()
    
    if save_dest:
        with open(os.path.join(save_dest, "all_metrics.json"), 'w') as fw:
            json.dump(all_subj_metrics, fw, indent=4)
    
        with open(os.path.join(save_dest, "all_suvr_diff.json"), 'w') as fw:
            json.dump(all_subj_suvr_diff, fw, indent=4)

    combined_df = pd.concat(all_suvr_df, ignore_index=True)
    combined_hemi_df = pd.concat(all_hemi_diff_df, ignore_index=True)
    return combined_df, combined_hemi_df, all_subj_metrics, all_subj_suvr_diff



def analyze_one_subj(subj, pet_types, save_dest, *test_dirs):
    '''
        Computes SUVR, PSNR, RMSE, MSSSIM, SUVR-asymmetry
        TODO: replace path please
    '''
    seg_file = os.path.join("Your path for segmentation", subj, "aseg.nii")
    roi_masks_global = utils.compute_masks(seg_file)
    roi_masks_left_right = utils.compute_hemispheric_masks(seg_file)
    pet_truth = sorted(glob.glob(os.path.join("Your path for acquired PET please", subj, 'reslice_PET_full.nii')))
    if len(pet_truth) == 0:
        pet_truth = sorted(glob.glob(os.path.join("Your path for acquired PET please", subj, 'reslice_PET_*.nii')))
        
    pet_truth = pet_truth[0]
    print(pet_truth)
    pets = {'truth': nib.load(pet_truth).get_fdata()}
    # idx = 1
    for i, directory in enumerate(test_dirs):
        pet_data = list(glob.glob(os.path.join(directory, subj) + "/*.nii"))
        if pet_data == []:
            print(f"Could not find predicted pet file for {subj}")
            continue
        curr_pred_pet = nib.load(pet_data[0]).get_fdata()
        if curr_pred_pet.shape != (256, 256, 89):
            padded = np.zeros((256, 256, 89))
            padded[:, :, 20:71] = curr_pred_pet
            curr_pred_pet = padded
        pets[pet_types[i]] = curr_pred_pet
        # fname = f'{save_dest}/{subj}_{pet_types[i]}_slice_visualization.png'
        # print("====Plotting Slice Visualization====")
        # plot_slice_visulization(subj, curr_pred_pet, pets['truth'], roi_masks, fname)

    suvr, suvr_diff = compute_suvr(roi_masks_global, 'cerebellum', pets, include_diff=True)
    all_metrics = compute_metrics(pets)
    suvr_hemi_diff, suvr_hemi_diff_diff = compute_suvr_asymmetry(roi_masks_left_right, roi_masks_global, pets)
    # print(suvr_hemi_diff)
    return suvr, suvr_diff, all_metrics, suvr_hemi_diff, suvr_hemi_diff_diff



if __name__ == "__main__":
    args = parser.parse_args()
    test_dir = args.dir
    exp_types = args.exp_types
    print("=====Starting=====")
    print(test_dir, exp_types)
    localtime = time.localtime(time.time())
    time_label = str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)
    if args.label == "":
        label = str(len(exp_types)) + "exps"
    else:
        label = args.label
    analysis_label = time_label + "_" + label

    dest = os.path.join(args.result, analysis_label)
    if not os.path.exists(dest):
        os.mkdir(dest)

    combined_suvr_df, combined_hemi_diff_df, all_subj_metrics, all_subj_suvr_diff = prepare_all_subjects(test_dir, exp_types, dest)

    for experiment in exp_types:
        global_bland_altman(combined_suvr_df, '_suvr', experiment, 'truth', utils.TARGET_ROIS, f"{dest}/{experiment}_vs_truth_SUVR.png")
        global_bland_altman(combined_hemi_diff_df, '_diff', experiment, 'truth', utils.LR_ROIS, f"{dest}/{experiment}_vs_truth_SUVR_Asymmetry.png")

    print("=====Done=====")