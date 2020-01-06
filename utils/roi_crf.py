import numpy as np
import cv2
import argparse
import random as rng

# crf postprocessing
from utils.crf import DenseCRF
from addict import Dict
import yaml
import os

home_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(home_dir, "config_crf.yaml")) as f:
	CRF_CONFIG = Dict(yaml.safe_load(f))

# log(softmax)
def roi_crf(img, softmax_pred, rois):
	postprocessor = DenseCRF(
		iter_max=CRF_CONFIG.CRF.ITER_MAX,
		pos_xy_std=CRF_CONFIG.CRF.POS_XY_STD,
		pos_w=CRF_CONFIG.CRF.POS_W,
		bi_xy_std=CRF_CONFIG.CRF.BI_XY_STD,
		bi_rgb_std=CRF_CONFIG.CRF.BI_RGB_STD,
		bi_w=CRF_CONFIG.CRF.BI_W,
	)
	rois_infer = []
	for roi in rois:
		xmin, ymin, xmax, ymax = roi[0], roi[1], roi[2], roi[3]
		prob_map = softmax_pred[:, ymin:ymax, xmin:xmax]
		roi  = img[ymin:ymax, xmin:xmax, :]
		roi = roi.astype(np.uint8)
		roi_crf_output = postprocessor(roi, prob_map)
		rois_infer.append(roi_crf_output)
	return rois_infer # np.array(rois_infer)

def img_crf(img, softmax_pred):
	postprocessor = DenseCRF(
		iter_max=CRF_CONFIG.CRF.ITER_MAX,
		pos_xy_std=CRF_CONFIG.CRF.POS_XY_STD,
		pos_w=CRF_CONFIG.CRF.POS_W,
		bi_xy_std=CRF_CONFIG.CRF.BI_XY_STD,
		bi_rgb_std=CRF_CONFIG.CRF.BI_RGB_STD,
		bi_w=CRF_CONFIG.CRF.BI_W,
	)
	img = img.astype(np.uint8)
	crf_output = postprocessor(img, softmax_pred)
	return crf_output

def get_roi_crf(img, softmax_pred, hard_pred):
	postprocessor = DenseCRF(
		iter_max=CRF_CONFIG.CRF.ITER_MAX,
		pos_xy_std=CRF_CONFIG.CRF.POS_XY_STD,
		pos_w=CRF_CONFIG.CRF.POS_W,
		bi_xy_std=CRF_CONFIG.CRF.BI_XY_STD,
		bi_rgb_std=CRF_CONFIG.CRF.BI_RGB_STD,
		bi_w=CRF_CONFIG.CRF.BI_W,
	)

	contours, _ = cv2.findContours(
					hard_pred.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	
	rois = []
	rois_crf = []
	if len(contours) > 1:
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
			if w*h <= 40*40:
				continue
			else:
				rect = get_rect_extend([x, y, x+w, y+h], img, 50)
				xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
				prob_map = softmax_pred[:, ymin:ymax, xmin:xmax]
				roi  = img[ymin:ymax, xmin:xmax, :]
				roi = roi.astype(np.uint8)
				roi_crf_output = postprocessor(roi, prob_map)
				rois.append([xmin, ymin, xmax, ymax])
				rois_crf.append(roi_crf_output)
	elif len(contours) == 0:
		rois.append([0, 0, img.shape[0], img.shape[1]])
		rois_crf.append(softmax_pred)
	else:
		rois = []
		x, y, w, h = cv2.boundingRect(contours[0])
		rect = get_rect_extend([x, y, x+w, y+h], img, 80)
		# rois.append([0, 0, img.shape[0], img.shape[1]])
		xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
		prob_map = softmax_pred[:, ymin:ymax, xmin:xmax]
		roi  = img[ymin:ymax, xmin:xmax, :]
		roi = roi.astype(np.uint8)
		roi_crf_output = postprocessor(roi, prob_map)
		rois_crf.append(roi_crf_output)
		rois.append(rect)
		# rois_crf = img_crf(img, softmax_pred)
	return rois_crf, rois

def get_rect_extend(rect, img, offset):
	im_h, im_w, _ = img.shape
	rect[0] = max(0, rect[0]-offset)
	rect[1] = max(0, rect[1]-offset)
	rect[2] = min(rect[2]+offset, im_w)
	rect[3] = min(rect[3]+offset, im_w)
	return rect










