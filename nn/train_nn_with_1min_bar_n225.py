import codecs as cd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential, clone_model, Model
from keras.layers.core import Dense, Activation, Dropout, Masking, Lambda
from keras.layers import InputLayer,LeakyReLU, CuDNNLSTM, Reshape, RepeatVector, TimeDistributed, Input, Flatten
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers, metrics, losses
from keras import backend as K

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import sys
sys.path.append('E:\\Users\\KNOL\\Documents\\python33\\my_lib\\')
from investment_data import (
    InvestmentData, sequential_time_series_to_separated,
    calc_rsi, calc_stoch,
)
import config

csv_list = []
for file_name in config.input_list:
    with cd.open(config.data_dir+file_name, "r", "UTF-8", "ignore") as csv_file:
        _df = pd.read_csv(csv_file, quotechar='"', header=0, index_col=0, parse_dates=True)    # convert data frame type by index_col
    csv_list.append(_df)

data = pd.concat(csv_list, axis=0, join='inner')

data = data[:50000]

#データの列名・行名の変更
data = data.rename(columns=config.renamed_columns)
data.index.name = config.renamed_index_name
data.index = data.index + data['Time'].apply(config.conv)

#時間足の変更
rule = config.bar_time
data["Open"] = data["Open"].resample(rule).first()
data["Close"] = data["Close"].resample(rule).last()
data["High"] = data["High"].resample(rule).max()
data["Low"] = data["Low"].resample(rule).min()
data["Volume"] = data["Volume"].resample(rule).sum()
data = data.dropna(how='any', axis=0)

#必要なテクニカルの計算
data['Average'] = data[['Open', 'High', 'Low', 'Close']].mean(axis='columns')
data['diff_Average'] = data['Average'].diff(1)
data['future_rsi'] = calc_rsi(data['Average'], 5).shift(-5)
data['stoch'] = calc_stoch(data['Average'], 5)
data['stoch_volume'] = calc_stoch(data['Volume'], 20)
data['shift_diff_average'] = data['diff_Average'].shift(-1)

data = data.dropna(how='any', axis=0)

input_data = data[config.input_columns].values
input_data = input_data - input_data.mean()
input_data = input_data/input_data.std()

#入力用に一定の時間区切りのデータに整形
timesteps = config.timesteps
X = sequential_time_series_to_separated(input_data, timesteps)
print(X[:, -1].mean())
Y = data[config.label_columns].values

X = X[timesteps:-1]
Y = Y[timesteps:-1]

print('\nX shape: ', X.shape, '\nY shape: ', Y.shape)

#学習データとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.1, shuffle=False, random_state = 0)

print('\nX train shape: ', X_train.shape, '\tY train shape: ', Y_train.shape)
print('\nX test shape: ', X_test.shape, '\tY test shape: ', Y_test.shape)

#こっから学習
input_shape = (X_train.shape[1], X_train.shape[2])
inputs = Input(shape=input_shape, name='input_layer')
flatten = Flatten()(inputs)
dropout = Dropout(0.5)(flatten)
z = Dense(256, activation='relu')(flatten)
z = Dropout(0.5)(z)
z = Dense(256, activation='relu')(z)
z = Dropout(0.5)(z)
outputs = Dense(Y_train.shape[1], activation='sigmoid', name='outputs')(z)
model = Model(inputs, outputs, name='model')

model.compile(
            loss='mse',
            optimizer=Adam(lr=0.001, decay=1e-10, amsgrad=True), #amsgradだとLoss下がった後の発散が抑えられてる
            metrics=['binary_accuracy'],
)

model.summary()

n_batch = 128
history = model.fit(X_train, Y_train,
          batch_size=n_batch,
          shuffle=True,
          epochs=1000,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.,  patience=100, verbose=0, mode='min')],
          verbose=2,
          validation_data=(X_test, Y_test)
          )

predicted = model.predict(X)
print(predicted.shape)

ax = np.arange(len(X))
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)

# fig.append_trace(go.Candlestick(
#     x=ax,
#     open=data['Open'],
#     high=data['High'],
#     low=data['Low'],
#     close=data['Close'],
#     name='OHLC'
# ), row=1, col=1)
#
# fig.update_layout(xaxis_rangeslider_visible=False)

fig.append_trace(go.Scatter(yaxis="y1",
                            x=ax,
                            y=X[:, -1, 0],
                            name='Average',
                            line=dict(color='red', width=1, dash="dashdot")
                            ), row=1, col=1)

fig.append_trace(go.Scatter(yaxis="y1",
                            x=ax,
                            y=Y[:, 0],
                            name='future_rsi',
                            line=dict(color='red', width=1)
                            ), row=2, col=1)

fig.append_trace(go.Scatter(yaxis="y1",
                            x=ax,
                            y=predicted[:, 0],
                            name='predict',
                            line=dict(color='blue', width=1)
                            ), row=2, col=1)

fig.show()