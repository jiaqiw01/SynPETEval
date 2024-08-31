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

import argparse

parser = argparse.ArgumentParser(prog='Roi Subject Analysis')
parser.add_argument('-d', '--dir', nargs='+', default=["/home/jiaqiw01/test_cases_fdg_1p_conditioned"])
parser.add_argument('-e', '--exp_types', nargs='+', default=["GAN_t1_t2f_pet_1p"])
parser.add_argument('-r', '--result', default="/home/jiaqiw01/test_roi_analysis")
parser.add_argument('-l', '--label', default="")


color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'lightblue', 'limegreen', 'black', 'maroon']

color_roi_mapping = {
    "cerebellum": '#1f77b4',
    "cortex": '#ff7f0e',
    "cerebral_white_matter": '#2ca02c',
    "hippocampus": '#d62728',
    "thalamus": 'maroon',
    "caudate": '#9467bd',
    "putamen": '#8c564b', 
    "brain_stem": '#e377c2',
    "globus_pallidus": '#7f7f7f',
    "amygdala": '#bcbd22',
    "CSF": '#17becf',
    "corpus_callosum": 'lightblue'
}


TARGET_ROIS = {
        "cerebellum": [7, 8, 46, 47],
        "cortex": [3, 42],
        "cerebral_white_matter": [2, 41],
        "thalamus": [10, 49],
        "hippocampus": [17, 53],
        "caudate": [11, 50],
        "putamen": [12, 51], 
        "brain_stem": [16],
        "globus_pallidus": [13, 52],
        "amygdala": [18, 54],
        "CSF": [24],
        "corpus_callosum": [251, 252, 253, 254, 255]
    } # cerebral cortex, cerebral wm, hippocampus, caudate, 

LR_ROIS = {
    "cerebellum": {'left': [7, 8], 
                   'right': [46, 47]},
    "cortex": {'left': [3],
              'right': [42]},
    "cerebral_white_matter": {'left': [2], 
                              'right':[41]},
    "hippocampus": {'left': [17],
                   'right': [53]},
    "caudate": {'left': [11],
               'right': [50]},
    "thalamus": {'left': [10],
                'right': [49]},
    "putamen": {'left': [12],
                'right': [51]}, 
    "globus_pallidus": {'left': [13],
                        'right': [52]},
    "amygdala": {'left': [18], 
                 'right': [54]},
}

Short_names = {
    "cerebellum": "cb",
    "cortex": 'cortex',
    "hippocampus": "hippo",
    "caudate": "caudate",
    "thalamus": "tha",
    "putamen": "putamen",
    "globus_pallidus": "gp",
    "amygdala": "amygdala"
}

def norm(img, norm_type='max'):
    if norm_type == 'max':
        img = (img - img.min()) / (img.max() - img.min())
        img = np.clip(img, a_min=0, a_max=1.0)
    elif norm_type == 'mean':
        img = img / img.mean()
    return img

def compute_msssim(img1, img2):
    '''
        More accurate SSIM evaluation than skimage
    '''
    ms_ssim_index = piq.multi_scale_ssim(img1, img2, data_range=1.)
    return ms_ssim_index

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
    
def compare_histograms(vol1, vol2, num_bins=128):
    '''
        Evaluate histogram with Wasserstein loss, can be a single slice or a whole volume
    '''
    hist1 = get_histogram(vol1, num_bins, output='all')[0]
    hist2 = get_histogram(vol2, num_bins, output='all')[0]
    # chi-square, may give nan value
    # chi_square, p_value = chisquare(hist1, hist2)
    # print(chi_square, p_value)
    # plt.subplot(121)
    # plt.bar(range(num_bins), hist1, width=3)
    # plt.subplot(122)
    # plt.bar(range(num_bins), hist2, width=3)
    # plt.show()
    # Wasserstein distance
    emd = wasserstein_distance(hist1, hist2)
    print(f'EMD: {emd}')


def compute_roi_mask(roi_volume_labels, label_vol, shape=(256, 256, 89)):
    roi_mask = np.zeros(shape)
    for lab in roi_volume_labels:
        m = np.where(label_vol==lab, 1, 0)
        roi_mask = m + roi_mask
    return roi_mask

def compute_masks(seg_file, threshold=900):
    '''
        Mask combine left & right hemisphere
    '''
    label_vol = nib.load(seg_file).get_fdata()

    target_volume_maps = {}
    for roi in TARGET_ROIS.keys():
        val = compute_roi_mask(TARGET_ROIS[roi], label_vol)
        target_volume_maps[roi] = val
    return target_volume_maps

def compute_hemispheric_masks(seg_file):
    '''
        Separate left & right mask
        return format: 
        {
            'cerebellum': {'left': mask1, 'right': mask2},
            'thalamus': {'left', ... 'right': ...},
            .....
        }
    '''
    label_vol = nib.load(seg_file).get_fdata()
    target_rois = {
            "cerebellum": {'left': [7, 8], 
                           'right': [46, 47]},
            "cortex": {'left': [3],
                      'right': [42]},
            "cerebral_white_matter": {'left': [2], 
                                      'right':[41]},
            "hippocampus": {'left': [17],
                           'right': [53]},
            "caudate": {'left': [11],
                       'right': [50]},
            "thalamus": {'left': [10],
                        'right': [49]},
            "putamen": {'left': [12],
                        'right': [51]}, 
            "globus_pallidus": {'left': [13],
                                'right': [52]},
            "amygdala": {'left': [18], 
                         'right': [54]},
        }
    roi_maps = {}
    for roi in target_rois.keys():
        roi_maps[roi] = {'left': compute_roi_mask(target_rois[roi]['left'], label_vol),
                        'right': compute_roi_mask(target_rois[roi]['right'], label_vol)}
    return roi_maps


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
        plot_asymmetry(hemi_diff, subj, axes_flatten_1[i])

        suvr_df = json_to_pd_row(subj, suvr)
        suvr_diff_df = json_to_pd_row(subj, suvr_diff)
        hemi_diff_df = json_to_pd_row(subj, hemi_diff)
    
        all_suvr_diff_df.append(suvr_diff_df)
        all_suvr_df.append(suvr_df)
        all_hemi_diff_df.append(hemi_diff_df)

        all_subj_metrics[subj] = all_metrics
        all_subj_suvr_diff[subj] = suvr_diff
        plot_suvr_diff_one_subj(suvr_diff, subj, axes_flatten_2[i], save_name=None)
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
            suvr_roi = suv_roi / (suv_ref + 1e-4)
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
                suvr_roi = suv_roi / (suv_ref + 1e-4)
                res[p][roi] = {'suvr': suvr_roi}
                if include_diff:
                    if p not in res_diff:
                        res_diff[p] = {}
                    res_diff[p][roi] = res['truth'][roi]['suvr'] - suvr_roi # how suvr deviates from the true pet
        return res, res_diff
    else:
        raise NotImplementedError
    
def compute_metrics(all_pets):
    res = {}
    truth = norm(all_pets['truth'][:, :, 20:71])
    for pet in all_pets.keys():
        if pet == 'truth':
            continue
        curr_pred_img = norm(all_pets[pet][:, :, 20:71])
        psnr = skimage.metrics.peak_signal_noise_ratio(truth, curr_pred_img, data_range=1)
        ssim = skimage.metrics.structural_similarity(truth, curr_pred_img, data_range=1)
        rmse = skimage.metrics.normalized_root_mse(truth, curr_pred_img)

        slice_true_torch = torch.Tensor(truth).unsqueeze(0).permute(3, 0, 1, 2)
        slice_pred_torch = torch.Tensor(curr_pred_img).unsqueeze(0).permute(3, 0, 1, 2)
        msssim = piq.multi_scale_ssim(slice_true_torch, slice_pred_torch, data_range=1.)
        res[pet] = {'psnr': psnr, 'rmse': rmse, 'ssim': ssim, 'msssim': msssim}
        print(f"PET: {pet}, PSNR: {psnr}, MSSSIM: {msssim}, RMSE: {rmse}, SSIM: {ssim}")
    return res


def compute_asymmetry(hemisphere_roi_masks, global_roi_masks, pet, include_diff=True, ref_name='cerebellum'):
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


def analyze_one_subj(subj, pet_types, save_dest, *test_dirs):
    '''
        Computes SUVR, PSNR, RMSE, MSSSIM, SUVR-asymmetry
    '''
    seg_file = os.path.join("/data/jiaqiw01/PET_MRI/data/segmentation", subj, "aseg.nii")
    roi_masks_global = compute_masks(seg_file)
    roi_masks_left_right = compute_hemispheric_masks(seg_file)
    pet_truth = sorted(glob.glob(os.path.join("/data/jiaqiw01/preprocessed_cases", subj, 'reslice_PET_full.nii')))
    if len(pet_truth) == 0:
        pet_truth = sorted(glob.glob(os.path.join("/data/jiahong/data/FDG_PET_preprocessed", subj, 'reslice_PET_*.nii')))
        
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
    suvr_hemi_diff, suvr_hemi_diff_diff = compute_asymmetry(roi_masks_left_right, roi_masks_global, pets)
    # print(suvr_hemi_diff)
    return suvr, suvr_diff, all_metrics, suvr_hemi_diff, suvr_hemi_diff_diff

def json_to_pd_row(subj, res):
    dfs = []
    for k, v in res.items():
        json_v = pd.json_normalize(v, sep='_')
        json_v['pet_type'] = k
        dfs.append(json_v)
    df = pd.concat(dfs, ignore_index=True)
    df['subject'] = subj
    col = df.pop("pet_type")
    df.insert(0, col.name, col)
    col = df.pop("subject")
    df.insert(0, col.name, col)
    return df


def plot_suvr_diff_one_subj(suvr_diff_dict, subj, ax, ignore_roi=['cerebellum'], save_name="test_suvr_diff_barplot.png"):
    for ignore in ignore_roi:
        if ignore in list(suvr_diff_dict.keys()):
            del suvr_diff_dict[ignore]
    diff_df = pd.DataFrame.from_dict(suvr_diff_dict).reset_index()
    pet_types = list(suvr_diff_dict.keys())
    diff_df_melted = pd.melt(diff_df, id_vars=['index'], value_vars=pet_types)
    sns.set_context("paper")
    sns.set_style("darkgrid")
    sns.barplot(diff_df_melted, x='index', y='value', hue='variable', ax=ax)
    ax.set_title(f"SUVR Difference Subject {subj}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=10)
    plt.tight_layout()
    # if save_name:
    #     plt.savefig(save_name, bbox_inches='tight', dpi=300)
    # else:
    #     plt.show()
    # return ax

def plot_slice_visulization(subj, pred, target, roi_masks, save, slice_idx=[40, 44, 48], target_rois=['cerebral_white_matter', 'cortex']):
    # idx = target.shape[-1]//2
    nrow = len(target_rois)
    ncol = len(slice_idx)
    print(nrow, ncol)
    fig, axes = plt.subplots(2, ncol, figsize=(12, 8))
    flatten_axes = axes.flatten()
    axes_idx = 0
    used_axes_idx = []
    for idx in slice_idx:
        pred_slice = pred[:, :, idx]
        target_slice = target[:, :, idx]
        dim_roi_mask_comb = np.zeros(target_slice.shape)
        for roi in target_rois:
            roi_mask_slice = roi_masks[roi][:, :, idx]
            dim_roi_mask = np.where(roi_mask_slice==1, 1, 0)
            dim_roi_mask_comb += dim_roi_mask

        target_img = dim_roi_mask_comb * target_slice 
        pred_img = dim_roi_mask_comb * pred_slice

        flatten_axes[axes_idx].imshow(pred_img, cmap='gray')
        flatten_axes[axes_idx].set_title(f"Slice {idx} Pred")
        flatten_axes[axes_idx].axis("off")
        used_axes_idx.append(axes_idx)

        flatten_axes[axes_idx + ncol].imshow(target_img, cmap='gray')
        flatten_axes[axes_idx + ncol].set_title(f"Slice {idx} Target")
        flatten_axes[axes_idx + ncol].axis("off")
        used_axes_idx.append(axes_idx + ncol)
        axes_idx += 1

    # for j in range(len(axes)):
    #     if j not in used_axes_idx:
    #         fig.delaxes(axes[j])
    rois = '_'.join(target_rois)
    fig.suptitle(f"{rois} Comparison Subject {subj}")
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')

def plot_asymmetry(info, subj, ax, save_name=None):
    sns.set_context("paper")
    sns.set_style("darkgrid")
    preds = list(info.keys())
    # print(preds)
    all_rois = list(LR_ROIS.keys())
    all_names = [Short_names[x] for x in all_rois]
    all_pred = [[] for _ in range(len(preds))]

    for i, pred in enumerate(preds):
        for roi in all_rois:
            pred_diff = info[pred][roi]['diff']
            all_pred[i].append(pred_diff)
    
    # print(all_rois, all_pred)

    bar_width = 0.8 / (len(preds)+1)
    r1 = np.arange(len(all_rois))
    
    # Creating the bar plot
    colors = color_list[:len(preds)]
    labels = preds
    
    for i in range(len(all_pred)):
        r = [x + bar_width * i for x in r1]
        ax.bar(r, all_pred[i], color=colors[i], width=bar_width, edgecolor='grey', label=labels[i])
    
    # Adding the labels
    ax.set_xlabel('ROI Name', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(all_rois))], all_names, rotation=30, fontsize=10)

    # Adding the legend
    ax.legend()
    ax.set_title(f"SUVR Asymmetry for Subject {subj}")
    # if save_name:
    #     plt.tight_layout()
    #     plt.savefig(save_name, bbox_inches='tight', dpi=400)
    # else:
    #     plt.show()
    # return fig


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
        global_bland_altman(combined_suvr_df, '_suvr', experiment, 'truth', TARGET_ROIS, f"{dest}/{experiment}_vs_truth_SUVR.png")
        global_bland_altman(combined_hemi_diff_df, '_diff', experiment, 'truth', LR_ROIS, f"{dest}/{experiment}_vs_truth_SUVR_Asymmetry.png")

    print("=====Done=====")