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
    "cerebral_white_matter": 'cwm',
    "cerebellum": "cb",
    "cortex": 'cortex',
    "hippocampus": "hippo",
    "caudate": "caudate",
    "thalamus": "tha",
    "putamen": "putamen",
    "globus_pallidus": "gp",
    "amygdala": "amygdala",
    "brain_stem": 'bs'
}
###### Helpers #######
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

def plot_suvr_diff_one_subj(suvr_diff_dict, subj, ax, ignore_roi=['cerebellum'], save_name=None):
    for ignore in ignore_roi:
        if ignore in list(suvr_diff_dict.keys()):
            del suvr_diff_dict[ignore]
    preds = list(suvr_diff_dict.keys())
    # print(preds)
    all_rois = list(TARGET_ROIS.keys())
    all_names = [Short_names[x] if x in Short_names else x for x in all_rois]
    all_pred = [[] for _ in range(len(preds))]
    
    for i, pred in enumerate(preds):
        for roi in all_rois:
            pred_diff = suvr_diff_dict[pred][roi]
            all_pred[i].append(pred_diff)
            
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
    ax.set_title(f"SUVR Difference for Subject {subj}")



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
    for j in range(len(axes)):
        if j not in used_axes_idx:
            fig.delaxes(axes[j])
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
    if save_name:
        plt.tight_layout()
        plt.savefig(save_name, bbox_inches='tight', dpi=400)
    else:
        plt.show()
    # return fig

