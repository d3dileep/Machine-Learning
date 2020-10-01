import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()
tf.disable_eager_execution()
tf.disable_v2_behavior()


file = sys.argv[1]
df = pd.read_csv(file)
date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
print(df.head())
file_path = '18.lstm_{}'.format(file)
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 50
dropout_rate = 0.9
future_day = 50

minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()

class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        seq_len,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        def global_pooling(x, func):
            batch_size = tf.shape(self.X)[0]
            num_units = x.get_shape().as_list()[-1]
            x = func(x, x.get_shape().as_list()[1], 1)
            x = tf.reshape(x, [batch_size, num_units])
            return x

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        drop = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cells, output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop,
            self.X,
            initial_state = self.hidden_layer,
            dtype = tf.float32,
            time_major = True,
        )
        self.outputs = self.outputs[:, :, 0]
        x = self.X
        masks = tf.sign(self.outputs)
        batch_size = tf.shape(self.X)[0]
        align = tf.matmul(self.X, tf.transpose(self.X, [0, 2, 1]))
        paddings = tf.fill(tf.shape(align), float('-inf'))
        k_masks = tf.tile(tf.expand_dims(masks, 1), [1, seq_len, 1])
        align = tf.where(tf.equal(k_masks, 0), paddings, align)
        align = tf.nn.tanh(align)
        q_masks = tf.to_float(masks)
        q_masks = tf.tile(tf.expand_dims(q_masks, -1), [1, 1, seq_len])
        align *= q_masks

        x = tf.matmul(align, x)
        g_max = global_pooling(x, tf.layers.max_pooling1d)
        g_avg = global_pooling(x, tf.layers.average_pooling1d)
        self.outputs = tf.concat([g_max, g_avg],1)
        self.logits = tf.layers.dense(self.outputs, output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
tf.reset_default_graph()
modelnn = Model(0.01, num_layers, df_log.shape[1], size_layer, df_log.shape[1], timestamp, dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    init_value = np.zeros((timestamp, num_layers * 2 * size_layer))
    total_loss = 0
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(
            df_log.iloc[k : k + timestamp].values, axis = 0
        )
        batch_y = df_log.iloc[k + 1 : k + timestamp + 1].values
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict = {
                modelnn.X: batch_x,
                modelnn.Y: batch_y,
                modelnn.hidden_layer: init_value,
            },
        )
        init_value = last_state
        total_loss += loss
    total_loss /= df_log.shape[0] // timestamp
    if (i + 1) % 100 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)

output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0] = df_log.iloc[0]
upper_b = (df_log.shape[0] // timestamp) * timestamp
init_value = np.zeros((timestamp, num_layers * 2 * size_layer))

for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[k : k + timestamp], axis = 0),
            modelnn.hidden_layer: init_value,
        },
    )
    output_predict[k + 1 : k + timestamp + 1] = out_logits
    init_value = last_state

df_log.loc[df_log.shape[0]] = out_logits[-1]
date_ori.append(date_ori[-1] + timedelta(days = 1))

for i in range(future_day - 1):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[-timestamp:], axis = 0),
            modelnn.hidden_layer: init_value,
        },
    )
    init_value = last_state
    output_predict[df_log.shape[0]] = out_logits[-1]
    df_log.loc[df_log.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(days = 1))

df_log = minmax.inverse_transform(df_log.values)
date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()


result = pd.DataFrame()

print(result.shape)
print(df.shape)
result['Date'] = df['Date']
result['True Close'] = df.iloc[:, 4]
result['Predicted Close'] = df_log[:df.shape[0], 3]
print(result.head())

result.to_csv(file_path,index=False)