import os
import sys
import math
import itertools

import numpy as np
import cv2
from joblib import Parallel,delayed
from natsort import natsorted
from loguru import logger
import cmap

from .openpose_detection import OpenposeDetection,SKEL25DEF

def get_assigned_pid(joint_id,joint_index,poses:dict):
    assigned_joints=[poses[pid][joint_id] for pid in poses]
    if joint_index not in assigned_joints:
        return -1
    assigned_pid=list(poses.keys())[assigned_joints.index(joint_index)]
    return assigned_pid

def check_pid_compatibility(assigned_pid,poses:dict)->bool:
    pa=poses[assigned_pid[0]]
    pb=poses[assigned_pid[1]]
    compatilibity=np.bitwise_and(
        np.array(pa)>-1,
        np.array(pb)>-1
    )
    if compatilibity.sum()==0:
        return True
    else:
        return False

# group pose
def group_bodypose(openpose_detection:OpenposeDetection,paf_thres=0.4):
    """
    associate pose with kruskal algorithm
    """
    candidates=[]
    for paf_id in range(len(openpose_detection.definition.paf_dict)):
        paf_pair=openpose_detection.definition.paf_dict[paf_id]
        jas=openpose_detection.joints[paf_pair[0]]
        jbs=openpose_detection.joints[paf_pair[1]]
        pafs=openpose_detection.pafs[paf_id]
        for ja_idx in range(len(jas)):
            for jb_idx in range(len(jbs)):
                paf_score=pafs[ja_idx][jb_idx]
                candidate=[paf_id,ja_idx,jb_idx,paf_score]
                candidates.append(candidate)
    # 筛选candidates
    candidates=natsorted(
        list(filter(lambda x:x[-1]>paf_thres,candidates)),
        key=lambda x:x[-1]
    )
    # group association
    pids=0
    poses=dict()
    for candidate in candidates:
        [paf_id,ja_idx,jb_idx,paf_score]=candidate
        paf_pair=openpose_detection.definition.paf_dict[paf_id]
        assigned_pid=[
            get_assigned_pid(paf_pair[0],ja_idx,poses),
            get_assigned_pid(paf_pair[1],jb_idx,poses)
        ]
        if assigned_pid[0]==-1 and assigned_pid[1]==-1:
            # 1. A & B not assigned yet -> create new person
            poses[pids]=[-1]*openpose_detection.definition.joint_size
            poses[pids][paf_pair[0]]=ja_idx
            poses[pids][paf_pair[1]]=jb_idx
            pids+=1
        elif assigned_pid[0]==-1 or assigned_pid[1]==-1:
            if assigned_pid[0]!=-1:
                # 2. A assigned but B not assigned -> merge B to A
                poses[assigned_pid[0]][paf_pair[1]]=jb_idx
            else:
                # 2. B assigend but A not assigned -> merge A to B
                poses[assigned_pid[1]][paf_pair[0]]=ja_idx
        elif assigned_pid[0]==assigned_pid[1]:
            # 3. A & B assigned to same person -> pass
            pass
        elif assigned_pid[0]!=assigned_pid[1]:
            # 4. A & B assigned to different person
            if check_pid_compatibility(assigned_pid,poses):
                # disjoint -> merge personB to personA
                for joint_id in range(openpose_detection.definition.joint_size):
                    if poses[assigned_pid[1]][joint_id]!=-1:
                        poses[assigned_pid[0]][joint_id]=poses[assigned_pid[1]][joint_id]
                poses.pop(assigned_pid[1])
            else:
                logger.info(f"ignore a connection with paf score: {paf_score}")
        else:
            raise NotImplementedError()
    # 筛选 poses
    persons=[]
    for pid in poses:
        pose=poses[pid]
        if (np.array(pose)>-1).sum()<4:
            continue
        persons.append(pose)
    return persons

# draw the body keypoint and lims
def draw_bodypose(img, poses:list,openpose_detection:OpenposeDetection):
    color_bar = cmap.Colormap(["red","blue","green"]) # 以RGBA顺序给出
    colors=np.flip(color_bar(np.linspace(0,1,len(poses)))[:,:3]*255,axis=1).astype(np.uint8) # 转换成BGR
    for color,pose in list(zip(colors,poses)):
        for paf_id in range(len(openpose_detection.definition.paf_dict)):
            paf_pair=openpose_detection.definition.paf_dict[paf_id]
            ja_index,jb_index=pose[paf_pair[0]],pose[paf_pair[1]]
            ja=openpose_detection.joints[paf_pair[0]][ja_index] if ja_index>-1 else None
            jb=openpose_detection.joints[paf_pair[1]][jb_index] if jb_index>-1 else None
            if ja is not None:
                img=cv2.circle(
                    img=img,
                    center=ja[:2].astype(np.int32),
                    radius=3,
                    color=(0,255,0),
                    thickness=-1
                )
            if jb is not None:
                img=cv2.circle(
                    img=img,
                    center=jb[:2].astype(np.int32),
                    radius=3,
                    color=(0,255,0),
                    thickness=-1
                )
            if ja is not None and jb is not None:
                img=cv2.line(
                    img=img,
                    pt1=ja[:2].astype(np.int32),
                    pt2=jb[:2].astype(np.int32),
                    color=color.tolist(),
                    thickness=2
                )
    return img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad
