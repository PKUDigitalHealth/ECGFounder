import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from net1d import Net1D
from util import save_checkpoint, my_eval, my_eval_with_dynamic_thresh, save_reg_checkpoint, quantile_accuracy
from finetune_model import ft_ECGFounder, ft_ramdom, ft_simCLR
from sklearn.model_selection import train_test_split
import torch.nn as nn
from net1d import Net1D
from torch.utils.data import Dataset, DataLoader
from dataset import LVEF_Dataset, CHD_Dataset, CKD_Dataset, age_Dataset, age_reg_Dataset, sex_Dataset, lab_Dataset, LVEF_reg_Dataset, lab_reg_Dataset

# Load the dataset

datasets = [ 'lab', 'age', 'LVEF']
Models = ['ECGFounder', 'ssl', 'random']

gpu_id = 1

### no need to change
batch_size = 512
lr = 1e-5
weight_decay = 1e-5
early_stop_lr = 1e-5
Epochs = 30
# eval_steps = 1000

### run_exp
base_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICS_txt/'
uddd_df = pd.read_csv('/hot_data/lijun/code/partners_ecg-master/d_icd_diagnoses.csv')


for run_id in datasets:
  for Model in Models:
    if run_id == 'LVEF':
        CustomDataset = LVEF_reg_Dataset
        df_label_path = '/home/lijun/code/LVEF.csv'
    elif run_id == 'lab':
        CustomDataset = lab_reg_Dataset
        df_label_path = '/data1/1shared/lijun/ecg/anyECG/data/lab_item/50963_NTproBNP_reg.csv'
    elif run_id == 'age':
        CustomDataset = age_reg_Dataset
        df_label_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICD_csv/age.csv'




    df_label = pd.read_csv(df_label_path)
    # Splitting the dataset into train, validation, and test sets

    train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

    train_dataset = CustomDataset(labels_df=train_df)
    val_dataset = CustomDataset(labels_df=val_df)
    test_dataset = CustomDataset(labels_df=test_df)

    # Example DataLoader usage
    trainloader = DataLoader(train_dataset, batch_size=256,num_workers=40, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=256,num_workers=40, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=256,num_workers=40, shuffle=False)

    tasks_file_path = os.path.join(base_path, 'class.txt')            
    tasks = []

    with open(tasks_file_path, 'r') as fin:
        for line in fin:
            tasks.append(line.strip())
    print(f"当前运行ID: {run_id}, 任务数: {len(tasks)}")

    saved_dir = '/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_reg_30epoch/{}'.format(run_id)
    print(saved_dir)
    print(df_label_path)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)
    copyfile('/data1/1shared/lijun/ecg/anyECG/code/MIMIC_finetune_template.py', os.path.join(saved_dir, 'MIMIC_finetune_template.py'))

    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    n_classes = len(tasks)

    if Model == 'ECGFounder':
        model = ft_ECGFounder(device, n_classes)
    elif Model == 'ssl':
        model = ft_simCLR(device, n_classes)
    elif Model == 'random':
        model = ft_ramdom(device, n_classes)

    criterion = nn.MSELoss()  

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

    ### train model
    best_mae = 100.
    step = 0
    current_lr = lr
    all_res = []
    pos_neg_counts = {}
    total_steps_per_epoch = len(trainloader)
    eval_steps = total_steps_per_epoch

    for epoch in range(Epochs):
        ### train
        for batch in tqdm(trainloader,desc='Training'):
            input_x, input_y = tuple(t.to(device) for t in batch)
            outputs = model(input_x)
            loss = criterion(outputs, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_steps == 0:

                # val
                model.eval()
                prog_iter_val = tqdm(valloader, desc="Validation", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_val):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        pred = model(input_x)
                        #pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        # print(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                df_pred_prob = pd.DataFrame(all_pred_prob)
                df_pred_prob.to_csv(f'/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_reg_30epoch/{run_id}/{Model}_pred.csv', index=False)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                df_all_gt = pd.DataFrame(all_gt)
                df_all_gt.to_csv(f'/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_reg_30epoch/{run_id}/{Model}_gt.csv', index=False)
                val_mae = np.mean(np.abs(all_pred_prob - all_gt))
                rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))


                quantiles = [0.25, 0.5, 0.75] 
                quantile_errors = quantile_accuracy(all_gt, all_pred_prob, quantiles)

                for q, error in quantile_errors.items():
                    print(f'Error at {q * 100}% quantile: {error}')

                print(f'MAE: {val_mae}')
                print(f'RMSE: {rmse}')

                # test
                model.eval()
                prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_test):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        pred = model(input_x)
                        #pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                mae = np.mean(np.abs(all_pred_prob - all_gt))
                rmse = np.sqrt(np.mean((all_pred_prob - all_gt) ** 2))

                ### save model and res
                is_best = bool(val_mae < best_mae)
                if is_best:
                    best_mae = val_mae
                    print('==> Saving a new val best!')
                    save_reg_checkpoint({
                        'epoch': epoch,
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'mae': val_mae,
                    }, saved_dir)
                current_lr = optimizer.param_groups[0]['lr']

                columns = ['mae', 'rmse']
                
                all_res.append([mae, rmse])
                df = pd.DataFrame(all_res, columns=columns)

                df.to_csv(os.path.join(saved_dir, f'res_{run_id}_{Model}.csv'), index=False, float_format='%.5f')
                
                scheduler.step(rmse)
                ### early stop
                current_lr = optimizer.param_groups[0]['lr']
                # if current_lr < early_stop_lr:
                #     print("Early stop")
                #     exit()
                    
                model.train() # set back to train









