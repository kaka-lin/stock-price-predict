import os
import csv
import errno
import numpy as np
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def file_processing(file_path, encode=None):
    data = []

    with open(file_path, encoding=encode) as file:
        rows = csv.reader(file, delimiter=",")
        n_row = 0

        for row in rows:
            if n_row != 0:
                #column -> 0: code, 1: date
                for column in range(2, len(row)):
                    data[n_row - 1].append(float(row[column].strip()))

            data.append([])
            n_row += 1

    del data[-1]
    return np.array(data)

def normalize_data(data, scaler, feature_len):
    minmaxscaler = scaler.fit(data)
    normalize_data = minmaxscaler.transform(data)

    return normalize_data

def inverse_normalize_data(data, scaler):
    for i in range(len(data)):
        data[i] = scaler.inverse_transform(data[i])

    return data

def generate_output(output, model_name, class_list):
    class_list = class_list
    _output = []

    for i in range(len(output)):
        _output.append([])
        _output[i].append(class_list[i])
        for j in range(len(output[i]) - 1):
            if output[i][j+1] > output[i][j]:
                _output[i].append(1)
                _output[i].append(output[i][j+1])
            elif output[i][j+1] == output[i][j]:
                _output[i].append(0)
                _output[i].append(output[i][j+1])
            else:
                _output[i].append(-1)
                _output[i].append(output[i][j+1])

    file_path = 'outputs/output_{}.csv'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(file_path, 'w+') as file:
        w = csv.writer(file)

        w.writerow(['ETFid','Mon_ud','Mon_cprice','Tue_ud','Tue_cprice','Wed_ud','Wed_cprice','Thu_ud','Thu_cprice','Fri_ud','Fri_cprice'])
        w.writerows(_output)

def plot_model_architecture(model, model_name):
    file_path = 'images/model/{}.png'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plot_model(model, to_file=file_path, show_shapes=True)

def save_model(model, model_name):
    file_path = 'model/{}.h5'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    model.save(file_path)

def plot_predict(data, data_predict, file_name):
    file_path = 'images/result/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)

    ax1.plot(data[:, 0, 3], color='black')
    ax1.plot(data_predict[:, 0, 3], color='red')
    ax1.title.set_text("Day 1")

    ax2.plot(data[:, 1, 3], color='black')
    ax2.plot(data_predict[:, 1, 3], color='red')
    ax2.title.set_text("Day 2")

    ax3.plot(data[:, 2, 3], color='black')
    ax3.plot(data_predict[:, 2, 3], color='red')
    ax3.title.set_text("Day 3")

    ax4.plot(data[:, 3, 3], color='black')
    ax4.plot(data_predict[:, 3, 3], color='red')
    ax4.title.set_text("Day 4")

    ax5.plot(data[:, 4, 3], color='black')
    ax5.plot(data_predict[:, 4, 3], color='red')
    ax5.title.set_text("Day 5")

    plt.savefig(file_path)
    #plt.show()

def plot_loss(history, file_name):
    file_path = 'images/loss/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(file_path)
    #plt.show()
