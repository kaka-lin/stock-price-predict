import math
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt

from utils import file_processing, data_processing

def load_data(data, time_step):
    target = data[0] # 標的: 0050

    seq_length = time_step + 1
    result = []
    for index in range(len(target) - seq_length):
        result.append(target[index: index + seq_length])

    result = np.array(result)

    train_size = int(len(result) * 0.67)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :-1]
    y_train = train[:, -1, -1]
    x_validate = validate[:, :-1]
    y_validate = validate[:, -1, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
    x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 5))

    return [x_train, y_train, x_validate, y_validate]

def build_model(units, input_shape=(None, None)):
    model = Sequential()

    model.add(LSTM(units=256,input_shape=input_shape,return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=256,return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(units=16, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(units=1, kernel_initializer="uniform", activation='linear'))

    return model

if __name__ == '__main__':
    class_data = file_processing('data/20180331/taetfp.csv', 'big5', 18)
    class_data = data_processing(class_data)

    time_step = 20
    batch_size = 64
    epochs = 100

    x_train, y_train, x_validate, y_validate = load_data(class_data, time_step=time_step)
    print('train: ', x_train.shape, y_train.shape)
    print('validate: ', x_validate.shape, y_validate.shape)

    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(x_train.shape[1], input_shape)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    plot_model(model, to_file='images/model.png', show_shapes=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    model.save('model/test_lstm.h5')

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.8f MSE (%.8f RMSE)' % (train_score, math.sqrt(train_score)))

    validate_score = model.evaluate(x_validate, y_validate, verbose=0)
    print('Test Score: %.8f MSE (%.8f RMSE)' % (validate_score, math.sqrt(validate_score)))

    train_predicted = model.predict(x_train, batch_size=batch_size)
    train_predicted = np.reshape(train_predicted, (train_predicted.size,))
    validate_predicted = model.predict(x_validate, batch_size=batch_size)
    validate_predicted = np.reshape(validate_predicted, (validate_predicted.size,))

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(y_train,color='black')
    plt.plot(train_predicted,color='red')

    plt.subplot(2,1,2)
    plt.plot(y_validate,color='black')
    plt.plot(validate_predicted,color='red')
    plt.savefig('images/result.png')

    plt.show()
