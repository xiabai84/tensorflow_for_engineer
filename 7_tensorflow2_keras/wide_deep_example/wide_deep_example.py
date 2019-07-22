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

# after normalization
# print(np.max(x_train_scaled), np.min(x_train_scaled))

# because we have different input shape, keras sequential can't take them directly
# but we want to put them in some way together, that's why we can only use 
# 1. function based API
# 2. subclass based API
#################### function based API ######################
'''
input = keras.layers.Input(shape = x_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
# now we got here input shape=(None, 8) and hidden2 shape=(None, 30) , they can't
# multiplicate with eachother yet.
# We should perform here a concate operation in order to merge to matrix together.
concat = keras.layers.concatenate([input, hidden2])  # shape=(None, 38)
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input], outputs = [output])
'''
###############################################################
#################### subclass based API ######################
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        # define model layers
        self.hidden1_layer = keras.layers.Dense(30, activation='relu')
        self.hidden2_layer = keras.layers.Dense(30, activation='relu')
        self.output_layer = keras.layers.Dense(1)
        
    def call(self, input):
        # overwrite model forward calculations
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input, hidden2])
        output = self.output_layer(concat)
        return output
        
model = WideDeepModel()
model.build(input_shape=(None, 8))
model.summary()

optimizer = keras.optimizers.SGD(3e-3)
# mean_squared_error make model as regression
model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-3)
]

history = model.fit(x_train_scaled, y_train, 
                    validation_data=(x_valid_scaled, y_valid),
                    epochs = 100, 
                    callbacks = callbacks)


def plot_learning_curves(history: History):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

plot_learning_curves(history)


test_loss, test_acc = model.evaluate(x_test_scaled, y_test)

# one_hot encoded results
predictions = model.predict(x_test_scaled)

index = 40

for indx in range(index):
    print(y_test[indx], predictions[indx])