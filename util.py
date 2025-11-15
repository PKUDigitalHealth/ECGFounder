import math
import os
import shutil
import h5py
import numpy as np
from time import gmtime, strftime
from matplotlib import pyplot as plt
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix, balanced_accuracy_score, roc_curve
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def filter_bandpass(signal, fs):
    """
    Bandpass filter
    :param signal: 2D numpy array of shape (channels, time)
    :param fs: sampling frequency
    :return: filtered signal
    """
    # Remove power-line interference
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])

    # Simple bandpass filter
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])

    # Remove baseline wander
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    filter_ecg = filtered_signal - baseline

    return filter_ecg

def my_eval_with_ci_thresh(
    gt, 
    pred, 
    n_bootstrap=1000, 
    ci_percentile=95
):
   
    n_task = gt.shape[1]
    
    rocaucs = []
    auprcs  = []
    ppvs    = []
    npvs    = []
    sensitivities = []
    specificities = []
    
    rocauc_cis = []
    auprc_cis  = []
    ppv_cis    = []
    npv_cis    = []
    sens_cis   = []
    spec_cis   = []

    for i in tqdm(range(n_task), desc="Evaluating tasks"):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        try:
            auroc_val = roc_auc_score(tmp_gt, tmp_pred)
        except ValueError:
            auroc_val = 0.0
        rocaucs.append(auroc_val)

        try:
            auprc_val = average_precision_score(tmp_gt, tmp_pred)
        except ValueError:
            auprc_val = 0.0
        auprcs.append(auprc_val)

        pred_labels = (tmp_pred > 0.5).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()

        if len(cm) == 1:
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:  # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # ============== 3) sensitivity (recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)

        # ============== 4) specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        # ============== 5) PPV (precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        ppvs.append(ppv)

        # ============== 6) NPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npvs.append(npv)

        rocauc_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='roc_auc',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        auprc_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='auprc',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        ppv_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='ppv',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        npv_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='npv',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        sens_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='sensitivity',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )
        spec_ci = bootstrap_ci(
            gt[:, i], pred[:, i], metric='specificity',
            n_bootstrap=n_bootstrap,
            ci_percentile=ci_percentile
        )

        rocauc_cis.append(rocauc_ci)
        auprc_cis.append(auprc_ci)
        ppv_cis.append(ppv_ci)
        npv_cis.append(npv_ci)
        sens_cis.append(sens_ci)
        spec_cis.append(spec_ci)

    rocaucs = np.array(rocaucs)
    auprcs  = np.array(auprcs)
    ppvs    = np.array(ppvs)
    npvs    = np.array(npvs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    mean_rocauc = np.mean(rocaucs)
    mean_auprc  = np.mean(auprcs)
    mean_ppv    = np.mean(ppvs)
    mean_npv    = np.mean(npvs)
    mean_sens   = np.mean(sensitivities)
    mean_spec   = np.mean(specificities)

    mean_metrics_dict = {
        "AUROC": mean_rocauc,
        "AUPRC": mean_auprc,
        "PPV": mean_ppv,
        "NPV": mean_npv,
        "Sensitivity": mean_sens,
        "Specificity": mean_spec
    }
    metrics_per_task_dict = {
        "AUROC": rocaucs,
        "AUPRC": auprcs,
        "PPV": ppvs,
        "NPV": npvs,
        "Sensitivity": sensitivities,
        "Specificity": specificities
    }
    ci_per_task_dict = {
        "AUROC": rocauc_cis,
        "AUPRC": auprc_cis,
        "PPV": ppv_cis,
        "NPV": npv_cis,
        "Sensitivity": sens_cis,
        "Specificity": spec_cis
    }

    return mean_metrics_dict, metrics_per_task_dict, ci_per_task_dict

def bootstrap_ci(
    gt, 
    pred, 
    metric, 
    n_bootstrap=1000, 
    ci_percentile=95
):
    """
    Calculates confidence intervals for a given metric using bootstrapping.

    Args:
        gt: Ground truth labels (numpy array), shape: [N,]
        pred: Prediction probabilities (numpy array), shape: [N,]
        metric: One of ['roc_auc', 'auprc', 'ppv', 'npv', 'sensitivity', 'specificity']
        n_bootstrap: Number of bootstrap samples to generate
        ci_percentile: Percentile for the confidence intervals

    Returns:
        (lower_bound, upper_bound): tuple of floats
    """
    from sklearn.metrics import (roc_auc_score, average_precision_score, 
                                 confusion_matrix, f1_score)

    n = len(gt)
    metrics_list = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(range(n), size=n, replace=True)
        gt_resampled = gt[indices]
        pred_resampled = pred[indices]

        if metric == 'roc_auc':
            try:
                val = roc_auc_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        elif metric == 'auprc':
            try:
                val = average_precision_score(gt_resampled, pred_resampled)
            except ValueError:
                val = 0.0

        else:
            pred_labels = (pred_resampled > 0.5).astype(int)
            cm = confusion_matrix(gt_resampled, pred_labels).ravel()
            if len(cm) == 1:
                if pred_labels.sum() == 0:  
                    tn, fp, fn, tp = cm[0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0]
            else:
                tn, fp, fn, tp = cm

            if metric == 'sensitivity':   # recall
                val = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == 'specificity':
                val = tn / (tn + fp) if (tn + fp) > 0 else 0
            elif metric == 'ppv':  # precision
                val = tp / (tp + fp) if (tp + fp) > 0 else 0
            elif metric == 'npv':
                val = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                val = 0.0

        metrics_list.append(val)

    alpha = (100 - ci_percentile) / 2
    lower_bound = np.percentile(metrics_list, alpha)
    upper_bound = np.percentile(metrics_list, 100 - alpha)
    return (lower_bound, upper_bound)

def quantile_accuracy(y_true, y_pred, quantiles):
    """
    :param y_true: 
    :param y_pred: 
    :param quantiles: e.g. [0.25, 0.5, 0.75]
    """
    quantile_errors = {}
    for q in quantiles:
        pred_quantile = np.percentile(y_pred, q * 100)
        true_quantile = np.percentile(y_true, q * 100)
        # calculate error
        quantile_errors[q] = abs(pred_quantile - true_quantile)
    
    return quantile_errors

def find_optimal_thresholds(gt, pred):
    optimal_thresholds = []
    for i in range(gt.shape[1]):
        fpr, tpr, thresholds = roc_curve(gt[:, i], pred[:, i])
        optimal_idx = np.argmax(tpr - fpr)  
        optimal_thresholds.append(thresholds[optimal_idx])
    return np.array(optimal_thresholds)


def my_eval_with_dynamic_thresh_and_roc(gt, pred, save_path=None):
    """
    Evaluates the model with dynamically adjusted thresholds for each task,
    and generates ROC curves with AUC values.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        - Overall mean of the metrics across all tasks
        - Per-metric mean across all tasks (as a list)
        - All metrics per task in a columnar format
        - ROC curves with AUC values and marked optimal operating points
    """
    optimal_thresholds = find_optimal_thresholds(gt, pred)
    n_task = gt.shape[1]
    rocaucs = []
    sensitivities = []
    specificities = []
    f1 = []
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(tmp_gt, tmp_pred)
            rocaucs.append(roc_auc)
        except:
            rocaucs.append(0.0)

        # Generate ROC curve
        fpr, tpr, thresholds = roc_curve(tmp_gt, tmp_pred)
        
        # Find the optimal threshold by minimizing distance to the top-left corner
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        # Plot ROC curve
        # plt.plot(fpr, tpr, label=f'Task {i+1} (AUC = {roc_auc:.2f})')
        # plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Point {i+1}', s=100)
        plt.plot(fpr, tpr, label=f'Task (AUC = {roc_auc:.3f})')
        plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Point', s=100)
        # plt.text(optimal_fpr, optimal_tpr, f'  (FPR={optimal_fpr:.2f}, TPR={optimal_tpr:.2f})', fontsize=10)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        f1s = f1_score(tmp_gt, pred_labels)
        f1.append(f1s)
    
    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves with AUC values and Optimal Operating Points')
    plt.legend(loc='lower right')
    plt.grid(False)
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1 = np.array(f1)
    
    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)

    return mean_rocauc, rocaucs, sensitivities, specificities, f1


def my_eval_with_dynamic_thresh(gt, pred):
    """
    Evaluates the model with dynamically adjusted thresholds for each task.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        - Overall mean of the metrics across all tasks
        - Per-metric mean across all tasks (as a list)
        - All metrics per task in a columnar format
    """
    optimal_thresholds = find_optimal_thresholds(gt, pred)
    n_task = gt.shape[1]
    rocaucs = []
    sensitivities = []
    specificities = []
    f1 = []
    auprcs = []  # Step 2: Initialize list for AUPRC

    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # AUPRC  # Step 3: Calculate AUPRC
        try:
            auprc = average_precision_score(tmp_gt, tmp_pred)
            auprcs.append(auprc)
        except:
            auprcs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
        # pred_labels = (tmp_pred > 0.5).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        f1s = f1_score(tmp_gt, pred_labels)
        f1.append(f1s)
    
    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1 = np.array(f1)
    auprcs = np.array(auprcs)  # Step 4: Compute mean AUPRC

    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)
    mean_auprc = np.mean(auprcs)  # Step 4: Compute mean AUPRC

    # Step 5: Update return statement
    return mean_rocauc, rocaucs, sensitivities, specificities, f1, auprcs, optimal_thresholds


def my_eval_new(gt, pred):
    thresh = 0.5
    n_task = gt.shape[1]
    res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 
        
        tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            tmp_res.append(roc_auc_score(tmp_gt, tmp_pred))
        except Exception as e:
            tmp_res.append(-1.0)
        

        res.append(tmp_res)
    
    res = np.array(res)
    return np.mean(res, axis=0), res[:,0]

def my_eval(gt, pred):
    """
    gt, pred are from multi-task
    
    Returns:
    - Overall mean of the metrics across all tasks
    - Per-metric mean across all tasks (as a list)
    - All metrics per task in a columnar format
    """
    thresh = 0.5

    n_task = gt.shape[1]
    # Initialize lists for each metric
    rocaucs = []
    sensitivities = []
    specificities = []
    for i in range(n_task):
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 
        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > thresh).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm
        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # Convert lists to numpy arrays for easier mean calculation and handling
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)

    return mean_rocauc, rocaucs, sensitivities, specificities


def my_eval_new_with_ci(gt, pred, n_bootstrap=10, ci=0.95):
    thresh = 0.5
    n_task = gt.shape[1]
    res = []
    ci_res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        
        # 
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 
        
        tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            auc_score = roc_auc_score(tmp_gt, tmp_pred)
            tmp_res.append(auc_score)
        except Exception as e:
            tmp_res.append(0.0)
        
        if tmp_res[0] != 0.0:
            lower_ci, upper_ci = bootstrap_ci(tmp_gt, tmp_pred, n_bootstrap=n_bootstrap, ci=ci)
            ci_res.append([lower_ci, upper_ci])
        else:
            ci_res.append([0.0, 0.0])

        res.append(tmp_res)
    
    res = np.array(res)
    ci_res = np.array(ci_res)
    return np.mean(res, axis=0), res[:, 0], ci_res

def get_time_str():
    return strftime("%Y%m%d_%H%M%S", gmtime())

def print_and_log(log_name, my_str):
    out = '{}|{}'.format(get_time_str(), my_str)
    print(out)
    with open(log_name, 'a') as f_log:
        print(out, file=f_log)

def save_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['val_auroc'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)

def save_reg_checkpoint(state, path):
    filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['step'], state['mae'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)


def find_optimal_threshold(gt, pred):
    n_task = gt.shape[1]
    optimal_thresholds = []

    for i in range(n_task):
        best_ba = -1  
        best_thresh = 0.5  
        for thresh in np.linspace(0.01, 0.99, 99):  
            pred_labels = (pred[:, i] > thresh).astype(int)
            ba = balanced_accuracy_score(gt[:, i], pred_labels)  
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

    return optimal_thresholds

def eval_with_dynamic_thresh(gt, pred):
    """
    Evaluates the model with dynamically adjusted thresholds for each task.

    Args:
        gt: Ground truth labels (numpy array)
        pred: Prediction probabilities (numpy array)

    Returns:
        - Overall mean of the metrics across all tasks
        - Per-metric mean across all tasks (as a list)
        - All metrics per task in a columnar format
    """
    optimal_thresholds = find_optimal_threshold(gt, pred)
    n_task = gt.shape[1]
    rocaucs = []
    sensitivities = []
    specificities = []
    f1 = []
    for i in range(n_task):
        tmp_gt = np.nan_to_num(gt[:, i], nan=0)
        tmp_pred = np.nan_to_num(pred[:, i], nan=0)

        # ROC-AUC
        try:
            rocaucs.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            rocaucs.append(0.0)

        # Sensitivity and Specificity
        pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
        cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
        # Handle different sizes of confusion matrix
        if len(cm) == 1:
            # Only one class present in predictions
            if pred_labels.sum() == 0:  # Only negative class predicted
                tn, fp, fn, tp = cm[0], 0, 0, 0
            else:                       # Only positive class predicted
                tn, fp, fn, tp = 0, 0, 0, cm[0]
        else:
            tn, fp, fn, tp = cm

        # Calculate Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivities.append(sensitivity)
        
        # Calculate Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

        f1s = f1_score(tmp_gt, pred_labels)
        f1.append(f1s)
    # Convert lists to numpy arrays
    rocaucs = np.array(rocaucs)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    f1 = np.array(f1)
    # Calculate means for each metric
    mean_rocauc = np.mean(rocaucs)

    return mean_rocauc, rocaucs, sensitivities, specificities, f1, optimal_thresholds
