from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from torchvision.transforms import ToTensor,Compose

def val(model,valoader,nclass=2,nogpu=False):
    model.eval()
    gts, preds = [], []
    for img_id, (img,gt_mask,_,gt_emask,_) in enumerate(valoader):
        # print(gt_mask.size(), gt_emask.size())
        gt_mask = gt_mask.numpy()[0]
        gt_emask = gt_emask.numpy()[0]
        if nogpu:
            img = Variable(img,volatile=True)
        else:
            img = Variable(img.cuda(),volatile=True)
        out_pred_map = model(img)
        # print("output: ", out_pred_map.size())
        # Get hard prediction
        if nogpu:
            soft_pred = out_pred_map.data.numpy()[0]
        else:
            soft_pred = out_pred_map.data.cpu().numpy()[0]

        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        # print(hard_pred.shape, gt_mask.shape)
        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_) 
    miou, _ = scores(gts, preds, n_class = nclass)
    return miou

def val_e(model,valoader,nclass=2,nogpu=False):
    model.eval()
    gts, preds, gts_e = [], [], []
    for img_id, (img,gt_mask,_,gt_emask,_) in enumerate(valoader):
        # print(gt_mask.size(), gt_emask.size())
        gt_mask = gt_mask.numpy()[0]
        gt_emask = gt_emask.numpy()[0]
        if nogpu:
            img = Variable(img,volatile=True)
        else:
            img = Variable(img.cuda(),volatile=True)
        out_pred_map = model(img)
        if nogpu:
            soft_pred = out_pred_map.data.numpy()[0]
        else:
            soft_pred = out_pred_map.data.cpu().numpy()[0]

        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        # print(hard_pred.shape, gt_mask.shape)
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        for gt_, gte_, pred_ in zip(gt_mask, gt_emask, hard_pred):
            gts.append(gt_)
            gts_e.append(gte_)
            preds.append(pred_) 
    miou, _ = scores(gts, preds, n_class = nclass)
    eiou, _ = scores(gts_e, preds, n_class = nclass)
    return miou, eiou


def val_e_sigmoid(model,valoader,nclass=2,nogpu=False):
    model.eval()
    gts, preds, gts_e = [], [], []
    for img_id, (img,gt_mask,_,gt_emask,_) in enumerate(valoader):
        # print(gt_mask.size(), gt_emask.size())
        gt_mask = gt_mask.numpy()[0]
        gt_emask = gt_emask.numpy()[0]
        if nogpu:
            img = Variable(img,volatile=True)
        else:
            img = Variable(img.cuda(),volatile=True)
        out_pred_map = model(img)
        out_pred_map = nn.Sigmoid()(out_pred_map)
        # print("output: ", out_pred_map.size())
        # Get hard prediction
        if nogpu:
            soft_pred = out_pred_map.data.numpy()[0]
        else:
            soft_pred = out_pred_map.data.cpu().numpy()[0]

        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        soft_pred[soft_pred>=0.5] = 1
        soft_pred[soft_pred<0.5] = 0
        hard_pred = soft_pred[0].astype(np.uint8)
        # print(hard_pred.shape, gt_mask.shape)
        # hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        for gt_, gte_, pred_ in zip(gt_mask, gt_emask, hard_pred):
            gts.append(gt_)
            gts_e.append(gte_)
            preds.append(pred_) 
    miou, _ = scores(gts, preds, n_class = nclass)
    eiou, _ = scores(gts_e, preds, n_class = nclass)
    return miou, eiou

