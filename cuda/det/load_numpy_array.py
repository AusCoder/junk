import numpy as np

array_path = "./mtcnn-output-arrays/stage-one/prob-0.npy"
arr = np.load(array_path)
print(arr.reshape(-1)[:10])
print(arr.shape)
