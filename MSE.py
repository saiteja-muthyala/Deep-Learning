from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target

# Handle missing values (if any)
data = data.dropna()

# Split the data into features and target
X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f'Linear Regression MSE: {mse_lin}')

from sklearn.ensemble import RandomForestRegressor

# Train the Random Forest model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=0)
rf_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_reg.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest Regressor MSE: {mse_rf}')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Configuration 1
model_1 = Sequential()
model_1.add(Dense(64, input_dim=X_train.shape[1], activation='sigmoid'))
model_1.add(Dense(32, activation='relu'))
model_1.add(Dense(1))

model_1.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model_1.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

y_pred_nn1 = model_1.predict(X_test)
mse_nn1 = mean_squared_error(y_test, y_pred_nn1)
print(f'Neural Network (Config 1) MSE: {mse_nn1}')

# Configuration 2
model_2 = Sequential()
model_2.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model_2.add(Dense(64, activation='relu'))
model_2.add(Dense(1))

model_2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model_2.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

y_pred_nn2 = model_2.predict(X_test)
mse_nn2 = mean_squared_error(y_test, y_pred_nn2)
print(f'Neural Network (Config 2) MSE: {mse_nn2}')

from sklearn.tree import DecisionTreeRegressor

# Train the Decision Tree model
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred_dt = dt_reg.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Decision Tree Regressor MSE: {mse_dt}')

print(f'Linear Regression MSE: {mse_lin}')
print(f'Random Forest Regressor MSE: {mse_rf}')
print(f'Neural Network (Config 1) MSE: {mse_nn1}')
print(f'Neural Network (Config 2) MSE: {mse_nn2}')
print(f'Decision Tree Regressor MSE: {mse_dt}')