import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model architecture
model = Sequential()
# Specify input shape using the Input layer
model.add(Input(shape=(32, 32, 3)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy')

# Define checkpoint callback
cb = ModelCheckpoint('myModel.keras', monitor='val_loss', save_best_only=True, mode="min")

# Calculate the accuracy on the test set before training
acc = model.evaluate(x_test, y_test)
print("Initial Accuracy:", acc)

# Train the model
result = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.2, callbacks=[cb])

# Plot the validation loss
plt.plot(result.history['val_loss'], label='val_loss')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()