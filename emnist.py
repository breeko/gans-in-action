import numpy as np
import pandas as pd

def process_emnist(arr, mappings):
    """ Process emnist letters by applying rotations and flipping """
    X = np.array(arr.iloc[:,1:]).reshape([-1,28,28])
    X = np.rot90(X, axes=(1,2))
    X = np.flip(X, axis=(1))
    y = np.array(arr.iloc[:,0])
    y = [mappings.get(code, code) for code in y]
    return X, y

def load_data():
	train = pd.read_csv("emnist-letters-train.csv", header=None)
	test = pd.read_csv("emnist-letters-test.csv", header=None)

	mappings = {}

	with open("emnist-letters-mapping.txt") as f:
		for line in f.readlines():
			code, lower, upper = line.split()
			mappings[int(code)] = chr(int(lower))

	X_train, y_train = process_emnist(train, mappings)
	X_test, y_test = process_emnist(test, mappings)

	return (X_train, y_train), (X_test, y_test)