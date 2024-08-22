import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
#Load the Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
print(x_train.shape,x_test.shape)
x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(type(y_train[0]))
print(x_train[0])
print(x_train.max())

# Normalize

x_test = x_test/255
x_train = x_train/255
print(x_test.max())
print(x_test.min())

#Model 1

model = Sequential()
model.add(Dense(10,input_dim = 784,activation = 'softmax'))
model.compile(optimizer = Adam(learning_rate=0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])

cb = ModelCheckpoint('myModel.keras',monitor = 'val_loss',save_best_only = True,mode="min")

result = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_split = 0.2,callbacks = [cb])
model.summary()
acc = model.evaluate(x_test,y_test)
print(acc)
print(acc[1])
print(result.history.keys())
plt.plot(result.history['loss'],label = 'val_loss')
plt.plot(result.history['val_loss'],label = 'train_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Model 2

model = Sequential()
model.add(Dense(10,input_dim = 784,activation = 'softmax'))
model.compile(optimizer = Adam(learning_rate=0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])

cb = ModelCheckpoint('myModel.keras',monitor = 'val_loss',save_best_only = True,mode="min")

result = model.fit(x_train,y_train,epochs = 2,batch_size = 32,validation_data = (x_test,y_test),callbacks = [cb])
model.summary()
acc = model.evaluate(x_test,y_test)
print(acc)
print(acc[1])
print(result.history.keys())
plt.plot(result.history['loss'],label = 'val_loss')
plt.plot(result.history['val_loss'],label = 'train_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Model 3

model.add(Dense(10,input_dim = 784,activation = 'softmax'))
model.compile(optimizer = Adam(learning_rate=0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])

cb = ModelCheckpoint('myModel.keras',monitor = 'val_loss',save_best_only = True,mode="min")

result = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_split = 0.2,callbacks = [cb])
model.summary()
acc = model.evaluate(x_test,y_test)
print(acc)
print(acc[1])
print(result.history.keys())
plt.plot(result.history['loss'],label = 'val_loss')
plt.plot(result.history['val_loss'],label = 'train_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Model 4

model.add(Dense(10,input_dim = 784,activation = 'softmax'))
model.compile(optimizer = Adam(learning_rate=0.1),loss = 'categorical_crossentropy',metrics = ['accuracy'])

cb = ModelCheckpoint('myModel.keras',monitor = 'val_loss',save_best_only = True,mode="min")

result = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_split = 0.2,callbacks = [cb])
model.summary()
acc = model.evaluate(x_test,y_test)
print(acc)
print(acc[1])
print(result.history.keys())
plt.plot(result.history['loss'],label = 'val_loss')
plt.plot(result.history['val_loss'],label = 'train_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Model 5


model.add(Dense(100,input_dim = 784,activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer = Adam(learning_rate=0.1),loss = 'categorical_crossentropy',metrics = ['accuracy'])

cb = ModelCheckpoint('myModel.keras',monitor = 'val_loss',save_best_only = True,mode="min")

result = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_split = 0.2,callbacks = [cb])
model.summary()
acc = model.evaluate(x_test,y_test)
print(acc)
print(acc[1])
print(result.history.keys())
plt.plot(result.history['loss'],label = 'val_loss')
plt.plot(result.history['val_loss'],label = 'train_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()