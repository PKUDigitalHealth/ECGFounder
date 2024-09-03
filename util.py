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

# 

def find_optimal_thresholds(gt, pred):
    optimal_thresholds = []
    for i in range(gt.shape[1]):
        fpr, tpr, thresholds = roc_curve(gt[:, i], pred[:, i])
        optimal_idx = np.argmax(tpr - fpr)  
        optimal_thresholds.append(thresholds[optimal_idx])
    return np.array(optimal_thresholds)

# def my_eval_with_dynamic_thresh_and_roc(gt, pred, model_name, save_path=None):
#     """
#     Evaluates the model with dynamically adjusted thresholds for each task,
#     and generates ROC curves with AUC values for multiple models.

#     Args:
#         gt: Ground truth labels (numpy array)
#         pred: Prediction probabilities (numpy array)
#         model_name: Name of the model being evaluated (string)
#         save_path: Path to save the combined ROC curves image (string)

#     Returns:
#         - Overall mean of the metrics across all tasks
#         - Per-metric mean across all tasks (as a list)
#         - All metrics per task in a columnar format
#         - ROC curves with AUC values and marked optimal operating points
#     """
#     optimal_thresholds = find_optimal_thresholds(gt, pred)
#     n_task = gt.shape[1]
#     rocaucs = []
#     sensitivities = []
#     specificities = []
#     f1 = []
    
#     for i in range(n_task):
#         tmp_gt = np.nan_to_num(gt[:, i], nan=0)
#         tmp_pred = np.nan_to_num(pred[:, i], nan=0)

#         # ROC-AUC
#         try:
#             roc_auc = roc_auc_score(tmp_gt, tmp_pred)
#             rocaucs.append(roc_auc)
#         except:
#             rocaucs.append(0.0)

#         # Generate ROC curve
#         fpr, tpr, thresholds = roc_curve(tmp_gt, tmp_pred)
        
#         # Find the optimal threshold by minimizing distance to the top-left corner
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#         optimal_fpr = fpr[optimal_idx]
#         optimal_tpr = tpr[optimal_idx]

#         # Plot ROC curve for the current model
#         plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
#         plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', s=100)

#         # Sensitivity and Specificity
#         pred_labels = (tmp_pred > optimal_thresholds[i]).astype(int)
#         cm = confusion_matrix(tmp_gt, pred_labels).ravel()
        
#         # Handle different sizes of confusion matrix
#         if len(cm) == 1:
#             # Only one class present in predictions
#             if pred_labels.sum() == 0:  # Only negative class predicted
#                 tn, fp, fn, tp = cm[0], 0, 0, 0
#             else:                       # Only positive class predicted
#                 tn, fp, fn, tp = 0, 0, 0, cm[0]
#         else:
#             tn, fp, fn, tp = cm

#         # Calculate Sensitivity (True Positive Rate)
#         sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#         sensitivities.append(sensitivity)
        
#         # Calculate Specificity (True Negative Rate)
#         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#         specificities.append(specificity)

#         f1s = f1_score(tmp_gt, pred_labels)
#         f1.append(f1s)
    
#     # Return metrics for the current model
#     rocaucs = np.array(rocaucs)
#     sensitivities = np.array(sensitivities)
#     specificities = np.array(specificities)
#     f1 = np.array(f1)
    
#     # Calculate means for each metric
#     mean_rocauc = np.mean(rocaucs)

#     return mean_rocauc, rocaucs, sensitivities, specificities, f1


# def find_optimal_thresholds(gt, pred):
#     """
#     Find optimal threshold for each task based on Balanced Accuracy.

#     Args:
#         gt: Ground truth labels (numpy array)
#         pred: Prediction probabilities (numpy array)

#     Returns:
#         optimal_thresholds: Optimal threshold for each task
#     """
#     n_task = gt.shape[1]
#     optimal_thresholds = []

#     for i in tqdm(range(n_task)):
#         best_ba = -1  
#         best_thresh = 0.5  
#         for thresh in np.linspace(0.01, 0.99, 99):  
#             pred_labels = (pred[:, i] > thresh).astype(int)
#             ba = balanced_accuracy_score(gt[:, i], pred_labels)  
#             if ba > best_ba:
#                 best_ba = ba
#                 best_thresh = thresh
#         optimal_thresholds.append(best_thresh)

#     return optimal_thresholds


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

    return mean_rocauc, rocaucs, sensitivities, specificities, f1


def my_eval_new(gt, pred):
    thresh = 0.5
    n_task = gt.shape[1]
    res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        
        # 检查NaN并替换
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 可以选择适合你情况的替换值
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 同上
        
        tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            tmp_res.append(roc_auc_score(tmp_gt, tmp_pred))
        except Exception as e:
            tmp_res.append(-1.0)
        
        try:
            tmp_res.append(average_precision_score(tmp_gt, tmp_pred))
        except Exception as e:
            tmp_res.append(-1.0)
        
        tmp_res.append(accuracy_score(tmp_gt, tmp_pred_binary))
        tmp_res.append(f1_score(tmp_gt, tmp_pred_binary))

        res.append(tmp_res)
    
    res = np.array(res)
    return np.mean(res, axis=0), res[:,0], res[:,1], res[:,2], res[:,3]

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
        tmp_gt = np.nan_to_num(tmp_gt, nan=0)  # 可以选择适合你情况的替换值
        tmp_pred = np.nan_to_num(tmp_pred, nan=0)  # 同上
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

def my_eval_binary(gt, pred):
    """
    Evaluate performance for a binary classification task.
    
    Parameters:
    - gt: Ground truth labels, expected shape (n_samples,)
    - pred: Predicted probabilities, expected shape (n_samples,)
    
    Returns:
    - ROC-AUC score for the binary classification task.
    """
    thresh = 0.5  # This threshold is mentioned but not used; consider using it if you're evaluating accuracy or another threshold-dependent metric.

    # Calculate ROC-AUC
    try:
        rocauc = roc_auc_score(gt, pred)
    except ValueError:
        rocauc = -1.0  # Handling cases where ROC-AUC cannot be calculated, such as if one class is missing in gt.

    return rocauc


def my_eval_old(gt, pred):
    """
    gt, pred are from multi-task

    rocauc, prauc, accuracy, f1
    """
    thresh = 0.5

    n_task = gt.shape[1]
    res = []

    for i in range(n_task):
        tmp_res = []
        tmp_gt = gt[:, i]
        tmp_pred = pred[:, i]
        #tmp_pred_binary = np.array(tmp_pred > thresh, dtype=float)

        try:
            tmp_res.append(roc_auc_score(tmp_gt, tmp_pred))
        except:
            tmp_res.append(0.0)
        try:
            tmp_res.append(average_precision_score(tmp_gt, tmp_pred))
        except:
            tmp_res.append(0.0)
        #tmp_res.append(accuracy_score(tmp_gt, tmp_pred_binary))
        #tmp_res.append(f1_score(tmp_gt, tmp_pred_binary))

        res.append(tmp_res)
    res = np.nan_to_num(np.array(res))

    return np.mean(res, axis=0), res[:,0], res[:,1]


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