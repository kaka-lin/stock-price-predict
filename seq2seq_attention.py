''' LSTM 預測未來5天
此為用 LSTM many-to-many 架構
預測未來5天的收盤價
'''
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape, Concatenate, Dot
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from utils import *

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred[:,:,3] - y_true[:,:,3]), axis=-1)

def load_data(data, time_step=20, after_day=1, validate_percent=0.67):
    seq_length = time_step + after_day
    result = []
    for index in range(len(data) - seq_length + 1):
        result.append(data[index: index + seq_length])

    result = np.array(result)
    print('total data: ', result.shape)

    train_size = int(len(result) * validate_percent)
    train = result[:train_size, :]
    validate = result[train_size:, :]

    x_train = train[:, :time_step]
    y_train = train[:, time_step:]
    x_validate = validate[:, :time_step]
    y_validate = validate[:, time_step:]

    return [x_train, y_train, x_validate, y_validate]


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def one_step_attention(a, s_prev, repeator, concatenator, densor, activator, dotor):
    s_prev = repeator(s_prev)
    concat = concatenator([s_prev, a])
    e = densor(concat)
    alphas = activator(e)
    context =  dotor([alphas, a])

    return context

def seq2seq_attention(feature_len=1, after_day=1, input_shape=(20, 1), time_step=20):
    # Define the inputs of your model with a shape (Tx, feature)
    X = Input(shape=input_shape)
    s0 = Input(shape=(100, ), name='s0')
    c0 = Input(shape=(100, ), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    all_outputs = []

    # Encoder: pre-attention LSTM
    encoder = LSTM(units=100, return_state=False, return_sequences=True, name='encoder')
    # Decoder: post-attention LSTM
    decoder = LSTM(units=100, return_state=True, name='decoder')
    # Output
    decoder_output = Dense(units=feature_len, activation='linear', name='output')
    model_output = Reshape((1, feature_len))

    # Attention
    repeator = RepeatVector(time_step)
    concatenator = Concatenate(axis=-1)
    densor = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights')
    dotor =  Dot(axes = 1)

    encoder_outputs = encoder(X)

    for t in range(after_day):
        context = one_step_attention(encoder_outputs, s, repeator, concatenator, densor, activator, dotor)

        a, s, c = decoder(context, initial_state=[s, c])

        outputs = decoder_output(a)
        outputs = model_output(outputs)
        all_outputs.append(outputs)

    all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=[X, s0, c0], outputs=all_outputs)

    return model

if __name__ == '__main__':
    class_list = ['50', '51', '52', '53', '54', '55', '56', '57', '58',
                  '59', '6201', '6203', '6204', '6208', '690', '692', '701', '713']

    scaler = MinMaxScaler(feature_range=(0, 1))

    validate_percent = 0.8
    time_step = 20
    after_day = 5
    batch_size = 64
    epochs = 300
    output = []

    model_name = sys.argv[0].replace(".py", "")

    for index in range(len(class_list)):
        _class = class_list[index]
        print('******************************************* class 00{} *******************************************'.format(_class))

        # read data from csv, return data: (Samples, feature)
        data = file_processing(
            'data/20180504_process/20180504_{}.csv'.format(_class))
        feature_len = data.shape[1]

        # normalize data
        data = normalize_data(data, scaler, feature_len)

        # test data
        x_test = data[-time_step:]
        x_test = np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))

        # get train and validate data
        x_train, y_train, x_validate, y_validate = load_data(
            data, time_step=time_step, after_day=after_day, validate_percent=validate_percent)

        print('train data: ', x_train.shape, y_train.shape)
        print('validate data: ', x_validate.shape, y_validate.shape)

        # model complie
        input_shape = (time_step, feature_len)
        model = seq2seq_attention(feature_len, after_day, input_shape, time_step)
        model.compile(loss=mean_squared_error, optimizer='adam')
        model.summary()
        plot_model_architecture(model, model_name=model_name)

        s0_train = np.zeros((x_train.shape[0],100))
        c0_train = np.zeros((x_train.shape[0],100))
        s0_validate = np.zeros((x_validate.shape[0],100))
        c0_validate = np.zeros((x_validate.shape[0],100))
        s0_test = np.zeros((x_test.shape[0],100))
        c0_test = np.zeros((x_test.shape[0],100))

        history = model.fit(
            [x_train, s0_train, c0_train], y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=([x_validate, s0_validate, c0_validate], y_validate))
        model_class_name = model_name + '_00{}'.format(_class)
        save_model(model, model_name=model_class_name)

        print('-' * 100)
        train_score = model.evaluate([x_train,s0_train,c0_train], y_train, batch_size=batch_size, verbose=0)
        print('Train Score: %.8f MSE (%.8f RMSE)' % (train_score, math.sqrt(train_score)))

        validate_score = model.evaluate([x_validate,s0_validate,c0_validate], y_validate, batch_size=batch_size, verbose=0)
        print('Test Score: %.8f MSE (%.8f RMSE)' % (validate_score, math.sqrt(validate_score)))

        train_predict = model.predict([x_train,s0_train,c0_train])
        validate_predict = model.predict([x_validate,s0_validate,c0_validate])
        test_predict = model.predict([x_test,s0_test,c0_test])

        # 回復預測資料值為原始數據的規模
        train_predict = inverse_normalize_data(train_predict, scaler)
        y_train = inverse_normalize_data(y_train, scaler)
        validate_predict = inverse_normalize_data(validate_predict, scaler)
        y_validate = inverse_normalize_data(y_validate, scaler)
        test_predict = inverse_normalize_data(test_predict, scaler)

        # 3 or 0: close 的位置, 0:5為五天
        ans = np.append(y_validate[-1, -1, 3], test_predict[-1, 0:5, 3])
        output.append(ans)
        #print("output: \n", output)

        # plot predict situation (save in images/result)
        file_name = 'result_' + model_name + '_00{}'.format(_class)
        plot_predict(y_validate, validate_predict, file_name=file_name)

        # plot loss (save in images/loss)
        file_name = 'loss_' + model_name + '_00{}'.format(_class)
        plot_loss(history, file_name)

    output = np.array(output)
    print(output)
    generate_output(output, model_name=model_name, class_list=class_list)
