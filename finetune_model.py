import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from net1d import Net1D
from resnet1d import ResNet1D

import torch.nn as nn
import torch


def ft_ECGFounder(device, n_classes):
  model = Net1D(
      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  checkpoint = torch.load('/hot_data/lijun/code/partners_ecg-master/checkpoint_10000_0.9204.pth', map_location=device)
  state_dict = checkpoint['state_dict']

  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  model.load_state_dict(state_dict, strict=False)

  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)

  model.to(device)

  return model

def ft_ramdom(device, n_classes):
  model = Net1D(
      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  model.to(device)   
  return model

def ft_simCLR(device, n_classes):
      model = Net1D(
      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

      checkpoint = torch.load('/data1/1shared/lijun/ecg/anyECG/code/checkpoint_30200_0.8296.pth', map_location=device)
      state_dict = checkpoint['state_dict']

      state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

      model.load_state_dict(state_dict, strict=False)

      model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)

      model.to(device)
      return model
    # model = ResNet1D(
    #     in_channels=1,
    #     base_filters=64, 
    #     kernel_size=16,
    #     stride=2,
    #     groups=32,
    #     n_block=48,
    #     n_classes=n_classes,
    #     downsample_gap=6,
    #     increasefilter_gap=12,
    #     use_do=True,
    # )

    # checkpoint = torch.load('/data1/1shared/lijun/ecg/ECG_SimCLR-master/saved_models/SimCLR_MIMICIV_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_99_260621.pt', map_location=device)
    # # print(checkpoint)
    # state_dict = checkpoint['scheduler_state_dict']

    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

    # model.load_state_dict(state_dict, strict=False)

    # model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)
    
    # model.to(device)

    # return model



