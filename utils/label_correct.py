import numpy as np
import random
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# using xgboost
import xgboost as xgb
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(64*64, 64*32), 
			nn.ReLU(True),
			nn.Linear(64*32, 64*16),
			nn.ReLU(True), 
			nn.Linear(64*16, 64*8))
		self.decoder = nn.Sequential(
			nn.Linear(64*8, 64*16),
			nn.ReLU(True),
			nn.Linear(64*16, 64*32),
			nn.ReLU(True),
			nn.Linear(64*32, 64),
			nn.Tanh())

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


def select_prototypes(gt, unc, unc_low_th=0.1, num_samples=768):
	unc_low = np.argwhere(unc <= unc_low_th)
	gt_corrosion_coord = np.argwhere(gt == 1.)
	gt_background_coord = np.argwhere(gt == 0)
	gt_corrosion_set = set([tuple(x) for x in gt_corrosion_coord])
	gt_background_set = set([tuple(x) for x in gt_background_coord])
	unc_low_set = set([tuple(x) for x in unc_low])
	# print("gt_corrosion_set: ", gt_corrosion_set)

	# get prototypes corrosion
	corrosion_prototypes_all = np.array([x for x in gt_corrosion_set & unc_low_set])
	background_prototypes_all = np.array([x for x in gt_background_set & unc_low_set])
	corrosion_ind = np.random.randint(corrosion_prototypes_all.shape[0], size=num_samples)
	background_ind = np.random.randint(background_prototypes_all.shape[0], size=num_samples)
	corrosion_prototypes = corrosion_prototypes_all[corrosion_ind]
	background_prototypes = background_prototypes_all[background_ind]

	return corrosion_prototypes, background_prototypes

def cosine_similarity(A, B):
	A = np.array(A)
	B = np.array(B)
	dot = np.dot(A, B.T)
	norma = np.linalg.norm(A)
	normb = np.linalg.norm(B, axis=1)
	cos = dot / np.dot(norma,  normb.T)
	return cos


def label_correct_similarity(gt, unc, prototypes_c, prototypes_b, features, unc_high_th=0.4, method='cos'):
	unc_high = np.argwhere(unc >= unc_high_th)
	unc_high_features = [features[x[0]][x[1]] for x in unc_high]

	proto_c_features = [features[x[0]][x[1]] for x in prototypes_c]
	proto_b_features = [features[x[0]][x[1]] for x in prototypes_b]

	similarities = []
	for coord, feature in zip(unc_high, unc_high_features):
		similarity_corrosion = cosine_similarity(feature, proto_c_features).mean()
		similarity_background = cosine_similarity(feature, proto_b_features).mean()
		if similarity_corrosion > similarity_background:
			gt[coord[0]][coord[1]] = 1
		else:
			gt[coord[0]][coord[1]] = 0
	return gt

def label_correct_kmeans(gt, unc, prototypes_c, prototypes_b, features, name, unc_high_th=0.4, method='cos', using_pca=False):
	unc_high = np.argwhere(unc >= unc_high_th)
	unc_high_features = np.array([features[x[0]][x[1]] for x in unc_high])
	pred_results = np.zeros(unc.shape)

	proto_c_features = [features[x[0]][x[1]] for x in prototypes_c]
	proto_b_features = [features[x[0]][x[1]] for x in prototypes_b]
	proto_features = np.concatenate((proto_c_features, proto_b_features), axis=0)
	if using_pca:
		pca = IncrementalPCA(n_components=2)
		features_reduce_train = pca.fit_transform(proto_features)
		proto_features = features_reduce_train
		features_reduce_test = pca.fit_transform(unc_high_features)
		unc_high_features = features_reduce_test


	if method == 'l2':
		kmeans = KMeans(2, random_state=0).fit(proto_features)
		pred_kmeans = kmeans.predict(unc_high_features)
		scatter_name = os.path.join('/home/yaok/software/ASSS/results', '{}_pca_pred.png'.format(name))
		c_ind = np.argwhere(pred_kmeans == 1)[:,0].astype(int)
		b_ind = np.argwhere(pred_kmeans == 0)[:,0].astype(int)
		plt.scatter(unc_high_features[c_ind][:,0], unc_high_features[c_ind][:,1], s=10, c='red')
		plt.scatter(unc_high_features[b_ind][:,0], unc_high_features[b_ind][:,1], s=10, c='blue')
		plt.savefig(scatter_name)
		plt.clf()
		for coord, pred in zip(unc_high, pred_kmeans):
			if gt[coord[0]][coord[1]] == 1 and pred == 0:
				gt[coord[0]][coord[1]] = 0
				pred_results[coord[0]][coord[1]] = 1
			if pred == 1:
				pred_results[coord[0]][coord[1]] = 2
	elif method == 'cos':
		proto_features_normalized = normalize(proto_features, axis=1, norm='l2')
		kmeans = KMeans(2, random_state=0).fit(proto_features_normalized)
		unc_high_features_normalized = normalize(unc_high_features, axis=1, norm='l2')
		pred_kmeans = kmeans.predict(unc_high_features_normalized)
		scatter_name = os.path.join('/home/yaok/software/ASSS/results', '{}_pca_pred.png'.format(name))
		c_ind = np.argwhere(pred_kmeans == 1)[:,0].astype(int)
		b_ind = np.argwhere(pred_kmeans == 0)[:,0].astype(int)
		plt.scatter(unc_high_features[c_ind][:,0], unc_high_features[c_ind][:,1], s=10, c='red')
		plt.scatter(unc_high_features[b_ind][:,0], unc_high_features[b_ind][:,1], s=10, c='blue')
		plt.savefig(scatter_name)
		plt.clf()
		for coord, pred in zip(unc_high, pred_kmeans):
			if gt[coord[0]][coord[1]] == 1 and pred == 0:
				gt[coord[0]][coord[1]] = 0
				pred_results[coord[0]][coord[1]] = 1
			if pred == 1:
				pred_results[coord[0]][coord[1]] = 2
	return gt, pred_results

def label_correct_kmeans_noprototype(gt, unc, features, name, unc_high_th=0.4, method='l2', using_pca=False):
	unc_high = np.argwhere(unc >= unc_high_th)
	unc_high_features = np.array([features[x[0]][x[1]] for x in unc_high])
	# print("unc_high_features: ", np.shape(unc_high_features))
	pred_results = np.zeros(unc.shape)

	if using_pca:
		pca = IncrementalPCA(n_components=2)
		features_reduce_test = pca.fit_transform(unc_high_features)
		unc_high_features = features_reduce_test

	if method == 'l2':
		kmeans = KMeans(2, random_state=0).fit(unc_high_features)
		pred_kmeans = kmeans.predict(unc_high_features)
		# clf = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(unc_high_features)
		# pred_kmeans = clf.predict(unc_high_features)
		scatter_name = os.path.join('/home/yaok/software/ASSS/results', '{}_pca_pred.png'.format(name))
		c_ind = np.argwhere(pred_kmeans == 1)[:,0].astype(int)
		b_ind = np.argwhere(pred_kmeans == 0)[:,0].astype(int)
		plt.scatter(unc_high_features[c_ind][:,0], unc_high_features[c_ind][:,1], s=10, c='red')
		plt.scatter(unc_high_features[b_ind][:,0], unc_high_features[b_ind][:,1], s=10, c='blue')
		plt.savefig(scatter_name)
		plt.clf()
		for coord, pred in zip(unc_high, pred_kmeans):
			if gt[coord[0]][coord[1]] == 1 and pred == 0:
				gt[coord[0]][coord[1]] = 0
				pred_results[coord[0]][coord[1]] = 1
			if pred == 1:
				pred_results[coord[0]][coord[1]] = 2
	elif method == 'cos':
		unc_high_features_normalized = normalize(unc_high_features, axis=1, norm='l2')
		kmeans = KMeans(2, random_state=0).fit(unc_high_features_normalized)
		pred_kmeans = kmeans.predict(unc_high_features_normalized)
		scatter_name = os.path.join('/home/yaok/software/ASSS/results', '{}_pca_pred.png'.format(name))
		c_ind = np.argwhere(pred_kmeans == 1)[:,0].astype(int)
		b_ind = np.argwhere(pred_kmeans == 0)[:,0].astype(int)
		plt.scatter(unc_high_features[c_ind][:,0], unc_high_features[c_ind][:,1], s=10, c='red')
		plt.scatter(unc_high_features[b_ind][:,0], unc_high_features[b_ind][:,1], s=10, c='blue')
		plt.savefig(scatter_name)
		plt.clf()
		for coord, pred in zip(unc_high, pred_kmeans):
			if gt[coord[0]][coord[1]] == 1 and pred == 0:
				gt[coord[0]][coord[1]] = 0
				pred_results[coord[0]][coord[1]] = 1
			if pred == 1:
				pred_results[coord[0]][coord[1]] = 2
	return gt, pred_results


def label_correct_xgboost(gt, unc, prototypes_c, prototypes_b, features, name, unc_high_th=0.4, using_pca=False):
	unc_high = np.argwhere(unc >= unc_high_th)
	unc_high_features = np.array([features[x[0]][x[1]] for x in unc_high])
	pred_results = np.zeros(unc.shape)

	proto_c_features = np.array([features[x[0]][x[1]] for x in prototypes_c])
	proto_b_features = np.array([features[x[0]][x[1]] for x in prototypes_b])
	proto_c_target = np.ones((proto_c_features.shape[0], 1))
	proto_b_target = np.zeros((proto_b_features.shape[0], 1))

	proto_features = np.concatenate((proto_c_features, proto_b_features), axis=0)
	proto_targets = np.concatenate((proto_c_target, proto_b_target), axis=0)
	proto_features, proto_targets = shuffle(proto_features, proto_targets)

	if using_pca:
		pca = IncrementalPCA(n_components=2)
		features_reduce_train = pca.fit_transform(proto_features)
		proto_features = features_reduce_train
		features_reduce_test = pca.fit_transform(unc_high_features)
		unc_high_features = features_reduce_test

	param_list = [
				("eta", 0.08), # learning_rate
				("max_depth", 6), 
				("subsample", 0.8), 
				("colsample_bytree", 0.8), 
				("objective", "multi:softmax"), 
				("eval_metric", "merror"), 
				("alpha", 8), 
				("lambda", 2), 
				("num_class", 2),
				# gpu support
				('gpu_id', 0), 
				('tree_method', 'gpu_hist')]

	sc = StandardScaler()
	proto_features_normalized = sc.fit_transform(proto_features)
	unc_high_features_normalized = sc.fit_transform(unc_high_features)

	X_train, X_valid, y_train, y_valid = train_test_split(proto_features_normalized, 
														proto_targets, test_size=0.1)
	n_rounds = 600
	early_stopping = 50
	d_train = xgb.DMatrix(X_train, label=y_train)
	d_val = xgb.DMatrix(X_valid, label=y_valid)
	eval_list = [(d_train, "train"), (d_val, "validation")]
	bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, 
		early_stopping_rounds=early_stopping, verbose_eval=True)

	pred_test = xgb.DMatrix(data=unc_high_features)
	pred_xgb = bst.predict(pred_test)
	print("pred_xgb: ", np.unique(pred_xgb))
	scatter_name = os.path.join('/home/yaok/software/ASSS/results', '{}_pca_pred.png'.format(name))
	c_ind = np.argwhere(pred_xgb == 1)[:,0].astype(int)
	b_ind = np.argwhere(pred_xgb == 0)[:,0].astype(int)
	plt.scatter(unc_high_features[c_ind][:,0], unc_high_features[c_ind][:,1], s=10, c='red')
	plt.scatter(unc_high_features[b_ind][:,0], unc_high_features[b_ind][:,1], s=10, c='blue')
	plt.savefig(scatter_name)
	plt.clf()
	for coord, pred in zip(unc_high, pred_xgb):
		if gt[coord[0]][coord[1]] == 1 and pred == 0:
			gt[coord[0]][coord[1]] = 0
			pred_results[coord[0]][coord[1]] = 1
		if pred == 1:
			pred_results[coord[0]][coord[1]] = 2
	return gt, pred_results

def label_correct_nn(gt, unc, prototypes_c, prototypes_b, features, name, unc_high_th=0.4):
	unc_high = np.argwhere(unc >= unc_high_th)
	unc_high_features = np.array([features[x[0]][x[1]] for x in unc_high])
	pred_results = np.zeros(unc.shape)

	proto_c_features = np.array([features[x[0]][x[1]] for x in prototypes_c])
	proto_b_features = np.array([features[x[0]][x[1]] for x in prototypes_b])
	proto_c_target = np.ones((proto_c_features.shape[0], 1))
	proto_b_target = np.zeros((proto_b_features.shape[0], 1))

	proto_features = np.concatenate((proto_c_features, proto_b_features), axis=0)
	proto_targets = np.concatenate((proto_c_target, proto_b_target), axis=0)
	sc = StandardScaler()
	proto_features_normalized = sc.fit_transform(proto_features)
	unc_high_features_normalized = sc.fit_transform(unc_high_features)
	train_data = np.concatenate((proto_features_normalized, proto_targets), axis=1)

	train_pixel = True
	if train_pixel:
		num_epochs = 500
		batch_size = 64
		learning_rate = 1e-5
		model = autoencoder().cuda()
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(
			model.parameters(), lr=learning_rate, weight_decay=1e-5)

		dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
		min_loss = 10000
		for epoch in range(num_epochs):
			model.train()
			for batch_data in dataloader:
				training_data, training_label = batch_data[:, :-1], batch_data[:, -1]
				training_data = training_data.contiguous().view(1, -1).type(torch.FloatTensor)
				training_label = training_label.type(torch.FloatTensor)
				training_data, training_label = Variable(training_data, ).cuda(), Variable(training_label).cuda()
				# print("training_data: ", training_data.size())
				output = model(training_data)
				loss = criterion(output, training_label)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print('epoch [{}/{}], loss:{:.4f}'
				.format(epoch + 1, num_epochs, loss.data))
			if loss.data < min_loss:
				min_loss = loss.data
				torch.save(model.state_dict(), 'pixel_pred.pth')
	else:
		model = autoencoder()
		model.load_state_dict(torch.load("/home/yaok/software/ASSS/pixel_pred.pth"))
		model = model.cuda()
		model.eval()
		num_samples = unc_high_features_normalized.shape[0]
		num_left = 64 - (num_samples % 64) if num_samples % 64 != 0 else 0
		print("num_samples: ", num_samples, " left: ", num_left)
		print(unc_high_features_normalized.shape, unc_high_features_normalized[0:num_left, :].shape)
		unc_high_features_normalized = np.concatenate((unc_high_features_normalized, 
					unc_high_features_normalized[0:num_left, :]), axis=0)
		unc_high = np.concatenate((unc_high, unc_high[0:num_left, :]), axis=0)
		num_samples = unc_high_features_normalized.shape[0]
		print("num_test: ", unc_high_features_normalized.shape, unc_high.shape)
		for i in range(0, num_samples, 64):
			coords, inputs = unc_high[i:(i+64),:], unc_high_features_normalized[i:(i+64),:]
			inputs_v = Variable(torch.from_numpy(inputs), requires_grad=False)
			inputs_v = inputs_v.contiguous().view(1, -1).type(torch.FloatTensor).cuda()
			outputs = model(inputs_v)
			outputs_np = outputs.data.cpu().numpy()
			outputs_np = outputs_np.reshape(-1, 1)
			preds = np.zeros(outputs_np.shape)
			preds[outputs_np >= 0.5] = 1
			print(coords.shape, preds.shape)
			for coord, pred in zip(coords, preds):
				if gt[coord[0]][coord[1]] == 1 and pred == 0:
					gt[coord[0]][coord[1]] = 0
					pred_results[coord[0]][coord[1]] = 1
				if pred == 1:
					pred_results[coord[0]][coord[1]] = 2
	return gt, pred_results





















