import csv
import math
import numpy as np
from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import file_processing_2, data_processing_2

def load_data(data, time_step, after_day=1, validate_percent=0.67):
    seq_length = time_step + after_day
    result = []
    for index in range( len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])

    result = np.array(result)
    print(result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, -1]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, -1]


    return [x_train, y_train, x_validate, y_validate]

def build_model(units, input_shape=(None, None)):
    model = Sequential()

    model.add(LSTM(units=50,input_shape=input_shape,return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(units=128,return_sequences=False))
    #model.add(Dropout(0.2))

    #model.add(Dense(units=16, activation='linear'))
    model.add(Dense(units=1, activation='linear'))

    return model


if __name__ == '__main__':
    scaler = MinMaxScaler(feature_range=(0, 1))

    _class = '52'
    time_step = 20
    validate_percent = 0.67
    batch_size = 64
    epochs = 300
    ans = []

    data = file_processing_2('data/0427_{}.csv'.format(_class))
    x_test = data[-time_step:]
    x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
    data = data_processing_2(data, scaler)

    for i in range(5):
        print('================================== day {} =================================='.format(i + 1))
        after_day = i + 1

        x_train, y_train, x_validate, y_validate = load_data(data, time_step=time_step, after_day=after_day, validate_percent=validate_percent)
        print('train: ', x_train.shape, y_train.shape)
        print('validate: ', x_validate.shape, y_validate.shape)

        print('last x data: \n', scaler.inverse_transform(x_validate[-1]))
        y = np.reshape(y_validate[-1], (1, 1))
        print('last y data: \n', scaler.inverse_transform(y))

        input_shape = (x_train.shape[1], x_train.shape[2])
        model = build_model(x_train.shape[1], input_shape)
        model.compile(loss='mse', optimizer='adam')
        #model.summary()
        plot_model(model, to_file='images/model.png', show_shapes=True)

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
        model.save('model/lstm_00{}_day{}.h5'.format(_class, i+1))

        train_score = model.evaluate(x_train, y_train, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (train_score, math.sqrt(train_score)))

        validate_score = model.evaluate(x_validate, y_validate, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (validate_score, math.sqrt(validate_score)))

        #model = load_model('model/lstm_0050_2.h5')

        train_predict = model.predict(x_train, batch_size=batch_size)
        validate_predict = model.predict(x_validate, batch_size=batch_size)
        test_predict = model.predict(x_test)

        # 回復預測資料值為原始數據的規模
        train_predict = scaler.inverse_transform(train_predict)
        y_train = scaler.inverse_transform(y_train)
        validate_predict = scaler.inverse_transform(validate_predict)
        y_validate = scaler.inverse_transform(y_validate)
        test_predict = scaler.inverse_transform(test_predict)

        plt.figure()

        plt.plot(y_validate,color='black')
        plt.plot(validate_predict,color='red')
        plt.savefig('images/result_00{}_day{}.png'.format(_class ,i+1))

        ans.append(test_predict)

        #plt.show()

    print(ans)

    with open('outputs/output_00{}.csv'.format(_class), 'w+') as file:
        w = csv.writer(file)

        w.writerows(ans)


    '''
    # 預測未來n天的股價
    n_day = 6
    ans = np.zeros((n_day, 1))
    test = x_validate[-1] # 最後n天
    print(scaler.inverse_transform(test), '\n')
    test = np.reshape(test, (1, test.shape[0], test.shape[1]))


    print('======================================================')

    for i in range(n_day):
        test_predict = model.predict(test)
        ans[i] = test_predict

        test = np.delete(test, 0, axis=1)
        test_predict = np.reshape(test_predict, (1, test_predict.shape[0], test_predict.shape[1]))
        test = np.concatenate((test, test_predict),axis=1)

    ans = scaler.inverse_transform(ans)
    print(ans)
    '''
