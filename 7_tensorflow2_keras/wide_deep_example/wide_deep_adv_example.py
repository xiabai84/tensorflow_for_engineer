import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.python.keras.callbacks import History

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Loading datasets
housing = fetch_california_housing()

# Split data into tran, test and validation
# by default train_test_split split data in 3: 1 -> train and test -> default param -> test_size = 0.25
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

# Data normalization
    # before normalization
print(np.max(x_train), np.min(x_train))

# perform normalization
scaler = StandardScaler()

# 1. data in x_train is int32, we need to convert them to float32 first 
# 2. convert x_train data from 
#    [None, 28, 28] -> [None, 784] 
#       -> after all reshape back to [None, 28, 28]
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled  = scaler.transform(x_test)

# in wide deep model we usualy use multiple inputs. for a simple example I just
# made from one dataset two subsets:
# the first one take first 6 columns and the second the last 6
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])

hidden1 = keras.layers.Dense(30, activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
# the same as multi-input, it is also possible to have a multi-output net
'''
output2 = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs = [input_wide, input_deep],
                           outputs = [output, output2])
'''
model = keras.models.Model(inputs = [input_wide, input_deep],
                           outputs = output)
model.summary()
# mean_squared_error make model as regression
model.compile(loss = "mean_squared_error", optimizer = "sgd", metrics = ["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)
]

x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]

history = model.fit([x_train_scaled_wide, x_train_scaled_deep], 
                    y_train, 
                    validation_data=([x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                    epochs = 100, 
                    callbacks = callbacks)


def plot_learning_curves(history: History):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)


test_loss, test_acc = model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test)

# one_hot encoded results
predictions = model.predict([x_test_scaled_wide, x_test_scaled_deep])

index = 40

for indx in range(index):
    print(y_test[indx], predictions[indx])