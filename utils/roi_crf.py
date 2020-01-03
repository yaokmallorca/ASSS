import numpy 
import cv2
import argparse
import random as rng

# crf
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# log(softmax)
def roi_crf(softmax_pred, img, rois, gt, infer_times=10):
	# roi size 60x60 [xmin, ymin, xmax, ymax]
	d = dcrf.DenseCRF2D(60, 60, 2)
	rois_infer = []
	for roi in rois:
		U = softmax_pred[ymin:ymax][xmin:xmax]
		roi_img = img[ymin:ymax][xmin:xmax]
		U = U.reshape(2, -1)
		# add Unary to crf
		d.setUnaryEnergy(U)

		# create the color-independent features and then add them to CRF
		feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
		d.addPairwiseEnergy(feats, compat=3,
							kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)
		feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
							img=img, chdim=2)
		d.addPairwiseEnergy(feats, compat=10,
							kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)
		Q = d.inference(infer_times)
		MAP = np.argmax(Q, axis=0)
		rois_infer.append(MAP.reshape(60, 60))
	return rois_infer

def roi_crf(softmax_pred, img, gt, infer_times=10):
	# roi size 60x60 [xmin, ymin, xmax, ymax]
	d = dcrf.DenseCRF2D(320, 320, 2)
	U = softmax_pred[ymin:ymax][xmin:xmax]
	roi_img = img[ymin:ymax][xmin:xmax]
	U = U.reshape(2, -1)
	# add Unary to crf
	d.setUnaryEnergy(U)

	# create the color-independent features and then add them to CRF
	feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
	d.addPairwiseEnergy(feats, compat=3,
						kernel=dcrf.DIAG_KERNEL,
						normalization=dcrf.NORMALIZE_SYMMETRIC)
	feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
						img=img, chdim=2)
	d.addPairwiseEnergy(feats, compat=10,
						kernel=dcrf.DIAG_KERNEL,
						normalization=dcrf.NORMALIZE_SYMMETRIC)
	Q = d.inference(infer_times)
	MAP = np.argmax(Q, axis=0)
	MAP = MAP.reshape(320, 320,)
	return rois_infer





