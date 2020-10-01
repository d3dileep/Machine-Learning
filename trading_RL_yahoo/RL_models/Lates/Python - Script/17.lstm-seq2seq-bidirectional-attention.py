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

num_layers = 1
size_layer = 128
timestamp = 5
epoch = 10
dropout_rate = 0.7
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
        forget_bias = 0.1,
        attention_size = 10,
    ):
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        backward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
        )
        forward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
        )
        self.X = tf.placeholder(tf.float32, [None, None, size])
        self.Y = tf.placeholder(tf.float32, [None, output_size])
        drop_backward = tf.nn.rnn_cell.DropoutWrapper(
            backward_rnn_cells, output_keep_prob = forget_bias
        )
        drop_forward = tf.nn.rnn_cell.DropoutWrapper(
            forward_rnn_cells, output_keep_prob = forget_bias
        )
        self.backward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * 2 * size_layer)
        )
        self.forward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * 2 * size_layer)
        )
        outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
            drop_forward,
            drop_backward,
            self.X,
            initial_state_fw = self.forward_hidden_layer,
            initial_state_bw = self.backward_hidden_layer,
            dtype = tf.float32,
        )
        outputs = list(outputs)
        attention_w = tf.get_variable(
            'attention_v1', [attention_size], tf.float32
        )
        query = tf.layers.dense(
            tf.expand_dims(last_state[0][:, size_layer:], 1), attention_size
        )
        keys = tf.layers.dense(outputs[0], attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.tanh(align)
        outputs[0] = tf.squeeze(
            tf.matmul(
                tf.transpose(outputs[0], [0, 2, 1]), tf.expand_dims(align, 2)
            ),
            2,
        )
        outputs[0] = tf.concat([outputs[0], last_state[0][:, size_layer:]], 1)

        attention_w = tf.get_variable(
            'attention_v2', [attention_size], tf.float32
        )
        query = tf.layers.dense(
            tf.expand_dims(last_state[1][:, size_layer:], 1), attention_size
        )
        keys = tf.layers.dense(outputs[1], attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.tanh(align)
        outputs[1] = tf.squeeze(
            tf.matmul(
                tf.transpose(outputs[1], [0, 2, 1]), tf.expand_dims(align, 2)
            ),
            2,
        )
        outputs[1] = tf.concat([outputs[1], last_state[1][:, size_layer:]], 1)

        with tf.variable_scope('decoder', reuse = False):
            self.backward_rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
            )
            self.forward_rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(num_layers)], state_is_tuple = False
            )
            backward_drop_dec = tf.nn.rnn_cell.DropoutWrapper(
                self.backward_rnn_cells_dec, output_keep_prob = forget_bias
            )
            forward_drop_dec = tf.nn.rnn_cell.DropoutWrapper(
                self.forward_rnn_cells_dec, output_keep_prob = forget_bias
            )
            self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(
                forward_drop_dec,
                backward_drop_dec,
                self.X,
                initial_state_fw = outputs[0],
                initial_state_bw = outputs[1],
                dtype = tf.float32,
            )
        self.outputs = list(self.outputs)
        attention_w = tf.get_variable(
            'attention_v3', [attention_size], tf.float32
        )
        query = tf.layers.dense(
            tf.expand_dims(self.last_state[0][:, size_layer:], 1),
            attention_size,
        )
        keys = tf.layers.dense(self.outputs[0], attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.tanh(align)
        self.outputs[0] = tf.squeeze(
            tf.matmul(
                tf.transpose(self.outputs[0], [0, 2, 1]),
                tf.expand_dims(align, 2),
            ),
            2,
        )

        attention_w = tf.get_variable(
            'attention_v4', [attention_size], tf.float32
        )
        query = tf.layers.dense(
            tf.expand_dims(self.last_state[1][:, size_layer:], 1),
            attention_size,
        )
        keys = tf.layers.dense(self.outputs[1], attention_size)
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.tanh(align)
        self.outputs[1] = tf.squeeze(
            tf.matmul(
                tf.transpose(self.outputs[1], [0, 2, 1]),
                tf.expand_dims(align, 2),
            ),
            2,
        )
        self.outputs = tf.concat(self.outputs, 1)
        self.logits = tf.layers.dense(self.outputs, output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)

tf.reset_default_graph()
modelnn = Model(0.01, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    init_value_forward = np.zeros((1, num_layers * 2 * size_layer))
    init_value_backward = np.zeros((1, num_layers * 2 * size_layer))
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
                modelnn.backward_hidden_layer: init_value_backward,
                modelnn.forward_hidden_layer: init_value_forward,
            },
        )
        loss = np.mean(loss)
        init_value_forward = last_state[0]
        init_value_backward = last_state[1]
        total_loss += loss
    total_loss /= df_log.shape[0] // timestamp
    if (i + 1) % 100 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)

output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0] = df_log.iloc[0]
upper_b = (df_log.shape[0] // timestamp) * timestamp
init_value_forward = np.zeros((1, num_layers * 2 * size_layer))
init_value_backward = np.zeros((1, num_layers * 2 * size_layer))
for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[k : k + timestamp], axis = 0),
            modelnn.backward_hidden_layer: init_value_backward,
            modelnn.forward_hidden_layer: init_value_forward,
        },
    )
    init_value_forward = last_state[0]
    init_value_backward = last_state[1]
    output_predict[k + 1 : k + timestamp + 1] = out_logits

out_logits, last_state = sess.run(
    [modelnn.logits, modelnn.last_state],
    feed_dict = {
        modelnn.X: np.expand_dims(df_log.iloc[upper_b:], axis = 0),
        modelnn.backward_hidden_layer: init_value_backward,
        modelnn.forward_hidden_layer: init_value_forward,
    },
)
init_value_forward = last_state[0]
init_value_backward = last_state[1]
output_predict[upper_b + 1 : df_log.shape[0] + 1] = out_logits
df_log.loc[df_log.shape[0]] = out_logits[-1]
date_ori.append(date_ori[-1] + timedelta(days = 1))

for i in range(future_day - 1):
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[-timestamp:], axis = 0),
            modelnn.backward_hidden_layer: init_value_backward,
            modelnn.forward_hidden_layer: init_value_forward,
        },
    )
    init_value_forward = last_state[0]
    init_value_backward = last_state[1]
    output_predict[df_log.shape[0]] = out_logits[-1]
    df_log.loc[df_log.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(days = 1))

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

df_log = minmax.inverse_transform(output_predict)
date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()


result = pd.DataFrame()

print(result.shape)
print(df.shape)
result['Date'] = df['Date']
result['True Close'] = df.iloc[:, 4]
result['Predicted Close'] = df_log[:df.shape[0], 3]
print(result.tail())

file_path = '17.lstm_{}'.format(file)

result.to_csv(file_path,index=False)
