import re
import csv
import decimal
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def file_processing(file_path, encode, num_etf):
    data = []

    # 每一個維度儲存一個標的的資訊
    for i in range(num_etf):
        data.append([])

    with open(file_path, encoding=encode) as file:
        rows = csv.reader(file, delimiter=",")
        n_row = 0
        prev_class = ''
        curr_class = '0050'
        n_class = 0

        for row in rows:
            if n_row != 0:
                prev_class = curr_class
                curr_class = row[0].strip()

                if curr_class != prev_class:
                    n_class += 1
                    for i in range(3, 8):
                        value = float(decimal.Decimal(re.sub(r'[^\d.]', '', row[i].strip())))
                        data[n_class].append(value)
                else:
                    for i in range(3, 8):
                        value = float(decimal.Decimal(re.sub(r'[^\d.]', '', row[i].strip())))
                        data[n_class].append(value)

            n_row += 1

    return np.array(data)

def data_processing(data):
    for i in range(len(data)):
        data[i] = np.reshape(data[i], (-1, 5))

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        Open = scaler.fit_transform(data[i][:, 0].reshape(-1,1))
        Open = np.reshape(Open, (1, -1))
        High = scaler.fit_transform(data[i][:, 1].reshape(-1,1))
        High = np.reshape(High, (1, -1))
        Low = scaler.fit_transform(data[i][:, 2].reshape(-1,1))
        Low = np.reshape(Low, (1, -1))
        Close = scaler.fit_transform(data[i][:, 3].reshape(-1,1))
        Close = np.reshape(Close, (1, -1))

        data[i][:, 0] = Open
        data[i][:, 1] = High
        data[i][:, 2] = Low
        data[i][:, 3] = Close

        # Changed data
        for n in range(len(data[i])):
            data[i][n][-2], data[i][n][-1] = data[i][n][-1], data[i][n][-2]

    return data
