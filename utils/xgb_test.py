import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
import os.path as osp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # z = (x - u) / s
from sklearn.model_selection import train_test_split
import xgboost as xgb

pwd = os.getcwd()
print("current path: ", pwd)
os.chdir(pwd)

def iris_test():
	iris = load_iris()
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	df['label'] = iris.target
	data = np.array(df.iloc[:100, [0, 1, -1]])
	X, y = data[:,:-1], data[:,-1]
	num_featues = len(X[0]) # 2
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=0
	)

	model = xgb.XGBClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))


def mnist_test()：
	from subprocess import check_output
	print(check_output(["ls", "./mnist_csv"]).decode("utf8"))
	train_set_path = osp.join(pwd, "mnist_csv/mnist_train.csv")
	test_set_path = osp.join(pwd, "mnist_csv/mnist_test.csv")
	train_df = pd.read_csv(train_set_path)
	test_df = pd.read_csv(test_set_path)
	print (train_df.shape, test_df.shape)

	sc = StandardScaler()
	X_std = sc.fit_transform(train_df.values[:, 1:])
	y = train_df.values[:, 0]

	test_std = sc.fit_transform(test_df.values[:, 1:])
	test_y = test_df.values[:, 0]

	print (X_std.shape, y.shape)
	print (test_std.shape)

	X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.1)
	print (X_train.shape, y_train.shape)
	print (X_valid.shape, y_valid.shape)

	param_list = [
				("eta", 0.08), 
				("max_depth", 6), 
				("subsample", 0.8), 
				("colsample_bytree", 0.8), 
				("objective", "multi:softmax"), 
				("eval_metric", "merror"), 
				("alpha", 8), 
				("lambda", 2), 
				("num_class", 10),
				# gpu support
				('gpu_id', 0), 
				('tree_method', 'gpu_hist')]
	n_rounds = 600
	early_stopping = 50

	d_train = xgb.DMatrix(X_train, label=y_train)
	d_val = xgb.DMatrix(X_valid, label=y_valid)
	eval_list = [(d_train, "train"), (d_val, "validation")]
	bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)
	bst.save_model('xgb.model')

	d_test = xgb.DMatrix(data=test_std)
	y_pred = bst.predict(d_test)
	accuracy = accuracy_score(test_y, y_pred)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

def corrosion_test()：