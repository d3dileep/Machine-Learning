import keras 
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
from datetime import datetime
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
np.random.seed(4)
tf.random.set_seed(4)
from util import csv_to_dataset, history_points

def data_split(symbol,days):
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(symbol+'.csv')

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

# define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    opt = optimizers.Adam(lr=0.0005)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=16, epochs=5, shuffle=True, validation_split=0.1)


    # evaluation

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict([ohlcv_histories, technical_indicators])
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    # print(scaled_mse)

    # plt.gcf().set_size_inches(22, 15, forward=True)

    # start = 0
    # end = -1

    # real = plt.plot(unscaled_y_test[start:end], label='real')
    # pred = plt.plot(y_test_predicted[start:end], label='predicted')

    model.save(f'technical_model.h5')