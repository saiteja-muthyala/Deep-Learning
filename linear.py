import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

#Generate Synthetic data

np.random.seed(42)
X = np.random.rand(100,1) # Data
y = (2 * X + np.random.rand(100,1)) * 2  # Label

#step - 1 : Define the model
model = Sequential()

# step - 2: Add the Layers
model.add(Dense(1,input_dim = 1))

# step - 3: Compile
model.compile(optimizer = SGD(learning_rate = 0.1),loss = 'mean_squared_error')

# step - 4: Train the model
model.fit(X,y, epochs = 100)

# make predictions
op = model.predict(X)
# Plot
mse = mean_squared_error(y, op)
print(mse)
plt.scatter(X,y,label = 'Original Data')
plt.plot(X,op,color = 'y',label = 'predicted data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.show()