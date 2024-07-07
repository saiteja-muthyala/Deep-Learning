import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
#Load the Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
'''print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)'''
print(x_train[0])
plt.imshow(x_train[59999])
plt.show()
print(np.max(x_train),np.max(x_test))
print(np.max(y_train),np.max(y_test))

  #  Normalization
# convert values to float
x_train.astype(float)
mean = np.mean(x_train)
