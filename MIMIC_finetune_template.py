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
from util import save_checkpoint, my_eval, my_eval_with_dynamic_thresh, my_eval_with_dynamic_thresh_and_roc
from finetune_model import ft_ECGFounder, ft_ramdom, ft_simCLR
from sklearn.model_selection import train_test_split
import torch.nn as nn
from net1d import Net1D
from torch.utils.data import Dataset, DataLoader
from dataset import LVEF_Dataset, CHD_Dataset, CKD_Dataset, age_Dataset, age_reg_Dataset, sex_Dataset, lab_Dataset, LVEF_reg_Dataset

# Load the dataset

datasets = [ 'lab', 'age', 'sex', 'CHD', 'CKD', 'LVEF']
Models = ['random', 'ssl', 'ECGFounder']

gpu_id = 0

### no need to change
batch_size = 512
lr = 5e-5
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
        CustomDataset = LVEF_Dataset
        df_label_path = '/home/lijun/code/LVEF.csv'
    elif run_id == 'lab':
        CustomDataset = lab_Dataset
        df_label_path = '/data1/1shared/lijun/ecg/anyECG/data/lab_item/50963_NTproBNP_N末端脑钠肽前体.csv'
    elif run_id == 'age':
        CustomDataset = age_Dataset
        df_label_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICD_csv/age_gender_ICD_I.csv'
    elif run_id == 'sex':
        CustomDataset = sex_Dataset
        df_label_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICD_csv/age_gender_ICD_I.csv'
    elif run_id == 'CHD':
        CustomDataset = CHD_Dataset
        df_label_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICD_csv/CHD.csv'
    elif run_id == 'CKD':
        CustomDataset = CKD_Dataset
        df_label_path = '/hot_data/lijun/code/partners_ecg-master/MIMIC/ICD_csv/age_ICD_N18.csv'



    df_label = pd.read_csv(df_label_path)
    # Splitting the dataset into train, validation, and test sets

    train_df, test_df = train_test_split(df_label, test_size=0.2, shuffle=False)
    #val_df, test_df = train_test_split(csv_file_path, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(labels_df=train_df)
    #val_dataset = CustomDataset(labels_df=val_df, data_dir=data_files_paths)
    test_dataset = CustomDataset(labels_df=test_df)

    # Example DataLoader usage
    trainloader = DataLoader(train_dataset, batch_size=256,num_workers=40, shuffle=True)
    #valloader = DataLoader(val_dataset, batch_size=350, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=256,num_workers=40, shuffle=False)

    tasks_file_path = os.path.join(base_path, 'class.txt')            
    tasks = []

    with open(tasks_file_path, 'r') as fin:
        for line in fin:
            tasks.append(line.strip())
    print(f"当前运行ID: {run_id}, 任务数: {len(tasks)}")

    saved_dir = '/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_30epoch/{}'.format(run_id)
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

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

    ### train model
    best_val_auroc = 0.
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
                prog_iter_val = tqdm(testloader, desc="Validation", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_val):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        logits = model(input_x)
                        pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        # print(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                df_pred_prob = pd.DataFrame(all_pred_prob)
                df_pred_prob.to_csv(f'/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_30epoch/{run_id}/{Model}_gt.csv', index=False)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                df_all_gt = pd.DataFrame(all_gt)
                df_all_gt.to_csv(f'/data1/1shared/lijun/ecg/anyECG/code/res/eval_all_30epoch/{run_id}/{Model}_pred.csv', index=False)

                res_val, res_val_auroc, res_test_sens, res_test_spec = my_eval(all_gt, all_pred_prob)
                val_auroc = res_val
                print('Epoch {} step {}, val: {:.4f}'.format(epoch, step, res_val))

                # test
                model.eval()
                prog_iter_test = tqdm(testloader, desc="Testing", leave=False)
                all_gt = []
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_test):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        logits = model(input_x)
                        pred = torch.sigmoid(logits)
                        all_pred_prob.append(pred.cpu().data.numpy())
                        all_gt.append(input_y.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_gt = np.concatenate(all_gt)
                all_gt = np.array(all_gt)
                # res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1 = my_eval_with_dynamic_thresh(all_gt, all_pred_prob)
                res_test, res_test_auroc, res_test_sens, res_test_spec, res_test_f1 = my_eval_with_dynamic_thresh_and_roc(all_gt, all_pred_prob, save_path=f'/data1/1shared/lijun/ecg/anyECG/code/res/auc_fig/{run_id}_{Model}.png')
                
                print('Epoch {} step {}, val: {:.4f}, test: {:.4f} '.format(epoch, step, res_val, res_test))

                ### save model and res
                is_best = bool(val_auroc > best_val_auroc)
                if is_best:
                    best_val_auroc = val_auroc
                    print('==> Saving a new val best!')
                    save_checkpoint({
                        'epoch': epoch,
                        'step': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'val_auroc': val_auroc,
                    }, saved_dir)
                current_lr = optimizer.param_groups[0]['lr']

                for i, task in enumerate(tasks):
                  pos_count = test_df[task].sum()
                  neg_count = len(test_df) - pos_count
                  all_res.append([task, res_test_auroc[i], res_test_sens[i], res_test_spec[i], res_test_f1[i], pos_count, neg_count])

                columns = ['Field_ID', 'AUROC', 'sensitivity', 'specificity', 'f1', 'pos_num','neg_num']
                
                
                df = pd.DataFrame(all_res, columns=columns)
                df['name'] = ''
                icd_to_title_map = uddd_df.set_index('icd_code')['long_title'].to_dict()

                df['name'] = df['Field_ID'].map(icd_to_title_map)

                df.to_csv(os.path.join(saved_dir, f'res_{run_id}_{Model}.csv'), index=False, float_format='%.5f')
                
                scheduler.step(val_auroc)
                ### early stop
                current_lr = optimizer.param_groups[0]['lr']
                # if current_lr < early_stop_lr:
                #     print("Early stop")
                #     exit()
                    
                model.train() # set back to train








