import os
import sys
import math

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from joblib import Parallel,delayed
import torchvision.transforms.functional
import torchvision.transforms.v2

from .util import padRightDownCorner
from .model import bodypose_25_model
from .openpose_detection import SKEL19DEF,SKEL25DEF,OpenposeDetection

class TorchOpenpose(nn.Module):
    def __init__(self,model_body25 = './weight/body_25.pth',paf_thres=0.4):
        super().__init__()

        self.model = bodypose_25_model()
        self.model.load_state_dict(torch.load(model_body25,weights_only=True))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        
        self.limbSeq = SKEL25DEF.paf_dict
        self.mapIdx = [
            [0,1], 
            [2,3],
            [4,5],
            [6,7],
            [8,9],
            [10,11],
            [12,13],
            [14,15],
            [16,17],
            [18,19],
            [20,21],
            [22,23],
            [24,25],
            [26,27],
            [28,29],
            [30,31],
            [32,33],
            [34,35],
            [36,37],
            [38,39],
            [40,41],
            [42,43],
            [44,45],
            [46,47],
            [48,49],
            [50,51]
        ]
        self.njoint = SKEL25DEF.joint_size + 1 # 最后一个是背景
        self.npaf = SKEL25DEF.paf_size*2
        self.maxpooling=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.paf_thres=paf_thres

    def get_heatmap_and_paf(self,m,multiplier,image,stride=8,padValue=128):
        """
        Openpose的网络模型真的很简单,首先是一个Backbone采用的是VGG19,VGG19会将原图下采样8倍得到特征图. stride=8,因为这里原图和特征图之前的尺寸相差8倍
        """
        channel_o,height_o,width_o=image.shape
        resize = torchvision.transforms.Resize([int(height_o*multiplier[m]),int(width_o*multiplier[m])]) 
        images_to_test=resize(image)   
        channel_t,height_t,width_t=images_to_test.shape
        pad = 4 * [None]
        pad[0] = 0 # left
        pad[1] = 0 # up
        pad[2] = 0 if (width_t % stride == 0) else stride - (width_t % stride) # right
        pad[3] = 0 if (height_t % stride == 0) else stride - (height_t % stride) # down
        pad=torchvision.transforms.Pad(
            padding=pad, # If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively,
            fill=padValue,
        )
        images_to_test_padded=pad(images_to_test)
        data=images_to_test_padded/256-0.5
        data=torch.unsqueeze(data,dim=0)
        with torch.no_grad():
            heatmap, paf = self.model(data)
        heatmap=heatmap.squeeze()
        paf=paf.squeeze()
        # extract outputs, resize, and remove padding
        channel,height,width=heatmap.shape
        resize = torchvision.transforms.Resize([height*stride,width*stride],interpolation=torchvision.transforms.InterpolationMode.BICUBIC) # 上采样8倍特征图
        heatmap = resize(heatmap)
        paf = resize(paf)
        heatmap = heatmap[:,:height_t,:width_t] # 去除padding
        paf = paf[:,:height_t,:width_t]
        resize = torchvision.transforms.Resize([height_o,width_o],interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        heatmap = resize(heatmap)
        paf = resize(paf)
        return heatmap,paf
    
    def get_peak(self,heatmap_avg,part,thre1):
        map_ori = heatmap_avg[part,:, :]
        # use maxpooling to get peak
        height,width=map_ori.shape
        tmp_heatmap = self.maxpooling(map_ori.unsqueeze(dim=0)).squeeze()
        peaks_binary = torch.logical_and(map_ori==tmp_heatmap, map_ori > thre1)
        peaks = torch.nonzero(peaks_binary).flip(dims=[1]) # note reverse
        peaks_with_score = [torch.concat([x,map_ori[x[1], x[0]].unsqueeze(dim=0)]) for x in peaks]
        return peaks_with_score
    
    def get_connection_candidate(self,paf_avg,k,all_peaks,mid_num):
        score_mid = paf_avg[self.mapIdx[k],:, :]
        candA = all_peaks[self.limbSeq[k][0]]
        candB = all_peaks[self.limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        # import pdb;pdb.set_trace()
        connection_candidate = []
        for i in range(nA):
            connections=[]
            for j in range(nB):
                vec = torch.subtract(candB[j][:2], candA[i][:2])
                norm = torch.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-10 # 防止为0
                vec = torch.divide(vec, norm)
                startend = torch.stack([torch.linspace(candA[i][0], candB[j][0], steps=mid_num), torch.linspace(candA[i][1], candB[j][1], steps=mid_num)]).T.long()
                vec_x = torch.stack([score_mid[0,startend[I][1], startend[I][0]] for I in range(len(startend))])
                vec_y = torch.stack([score_mid[1,startend[I][1], startend[I][0]] for I in range(len(startend))])
                score_midpts = torch.multiply(vec_x, vec[0]) + torch.multiply(vec_y, vec[1]) # 两个单位向量的点乘
                score_paf=score_midpts.mean()
                score_paf=torch.clip(score_paf,0,1) # 限定 paf score 的范围为 0~1
                score_paf=torch.where(score_paf>self.paf_thres,score_paf,torch.zeros_like(score_paf,device="cuda"))
                connections.append(score_paf.item())
            connection_candidate.append(connections)
        return connection_candidate

    def forward(self, image_path,scale_search = [0.7, 1.0, 1.3],definition=SKEL25DEF): # # issues this scale search option is better
        boxsize = 368 # base resolution
        stride = 8
        padValue = 128
        thre1 = 0.1
        if isinstance(image_path,str):
            image=torchvision.io.read_image(image_path)
        else:
            image=torch.from_numpy(image_path)
            image=image.permute(2,0,1) # H,W,C -> C,H,W
            image=image.flip(dims=[0]) # BGR -> RGB
        if torch.cuda.is_available():
            image=image.cuda()
        channel,height,width=image.shape
        
        multiplier = [x * boxsize / height for x in scale_search]

        heatmap_avg=torch.zeros((self.njoint,height,width),dtype=torch.float32,device="cuda")
        paf_avg=torch.zeros((self.npaf,height,width),dtype=torch.float32,device="cuda")
        for m in range(len(multiplier)):
            heatmap,paf=self.get_heatmap_and_paf(
                m,multiplier,image,stride,padValue
            )
            heatmap_avg+=heatmap/len(multiplier)
            paf_avg+=paf/len(multiplier)
        gaussian = torchvision.transforms.GaussianBlur(kernel_size=11,sigma=3) # scipy gaussian filter 默认kernel_size=11
        heatmap_avg=gaussian(heatmap_avg)
        
        all_peaks=[]
        for part in range(self.njoint - 1): # 最后一个维度为背景
            all_peak=self.get_peak(heatmap_avg,part,thre1)
            all_peaks.append(all_peak)

        mid_num = 10 # 积分的采样数

        all_connections=[]
        for k in range(len(self.mapIdx)):
            all_connection=self.get_connection_candidate(paf_avg,k,all_peaks,mid_num)
            all_connections.append(all_connection)
        # download all from gpu
        for part in range(self.njoint - 1):
            for i in range(len(all_peaks[part])):
                all_peaks[part][i]=all_peaks[part][i].cpu().numpy()
        
        openpose_detection=OpenposeDetection(definition=definition)
        openpose_detection.joints=all_peaks
        openpose_detection.pafs=all_connections
        if definition is SKEL25DEF:
            pass
        elif definition is SKEL19DEF:
            openpose_detection.mapping()
        else:
            raise NotImplementedError()
        return openpose_detection
