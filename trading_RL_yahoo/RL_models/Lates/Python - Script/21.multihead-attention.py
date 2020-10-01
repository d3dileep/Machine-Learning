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

timestamp = 5
epoch = 50
future_day = 50

minmax = MinMaxScaler().fit(df.iloc[:, 1:].astype('float32'))
df_log = minmax.transform(df.iloc[:, 1:].astype('float32'))
df_log = pd.DataFrame(df_log)
df_log.head()

def embed_seq(
    inputs, vocab_size = None, embed_dim = None, zero_pad = False, scale = False
):
    lookup_table = tf.get_variable(
        'lookup_table', dtype = tf.float32, shape = [vocab_size, embed_dim]
    )
    if zero_pad:
        lookup_table = tf.concat(
            (tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis = 0
        )
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (embed_dim ** 0.5)
    return outputs


def learned_positional_encoding(
    inputs, embed_dim, zero_pad = False, scale = False
):
    T = inputs.get_shape().as_list()[1]
    outputs = tf.range(T)
    outputs = tf.expand_dims(outputs, 0)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])
    return embed_seq(outputs, T, embed_dim, zero_pad = zero_pad, scale = scale)


def layer_norm(inputs, epsilon = 1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims = True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(
        'gamma', params_shape, tf.float32, tf.ones_initializer()
    )
    beta = tf.get_variable(
        'beta', params_shape, tf.float32, tf.zeros_initializer()
    )
    return gamma * normalized + beta


def pointwise_feedforward(inputs, num_units = [None, None], activation = None):
    outputs = tf.layers.conv1d(
        inputs, num_units[0], kernel_size = 1, activation = activation
    )
    outputs = tf.layers.conv1d(
        outputs, num_units[1], kernel_size = 1, activation = None
    )
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


class Model:
    def __init__(
        self,
        dimension_input,
        dimension_output,
        seq_len,
        learning_rate,
        num_heads = 8,
        attn_windows = range(1, 6),
    ):
        self.size_layer = dimension_input
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.X = tf.placeholder(tf.float32, [None, seq_len, dimension_input])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        feed = self.X
        for i, win_size in enumerate(attn_windows):
            with tf.variable_scope('attn_masked_window_%d' % win_size):
                feed = self.multihead_attn(feed, self.window_mask(win_size))
        feed += learned_positional_encoding(feed, dimension_input)
        with tf.variable_scope('multihead'):
            feed = self.multihead_attn(feed, None)
        with tf.variable_scope('pointwise'):
            feed = pointwise_feedforward(
                feed,
                num_units = [4 * dimension_input, dimension_input],
                activation = tf.nn.relu,
            )
        self.logits = tf.layers.dense(feed, dimension_output)[:, -1]
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)
        self.correct_pred = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)
        )
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def multihead_attn(self, inputs, masks):
        T_q = T_k = inputs.get_shape().as_list()[1]
        Q_K_V = tf.layers.dense(inputs, 3 * self.size_layer, tf.nn.relu)
        Q, K, V = tf.split(Q_K_V, 3, -1)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis = 2), axis = 0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis = 2), axis = 0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis = 2), axis = 0)
        align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        align = align / np.sqrt(K_.get_shape().as_list()[-1])
        if masks is not None:
            paddings = tf.fill(tf.shape(align), float('-inf'))
            align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        outputs = tf.matmul(align, V_)
        outputs = tf.concat(
            tf.split(outputs, self.num_heads, axis = 0), axis = 2
        )
        outputs += inputs
        return layer_norm(outputs)

    def window_mask(self, h_w):
        masks = np.zeros([self.seq_len, self.seq_len])
        for i in range(self.seq_len):
            if i < h_w:
                masks[i, : i + h_w + 1] = 1.0
            elif i > self.seq_len - h_w - 1:
                masks[i, i - h_w :] = 1.0
            else:
                masks[i, i - h_w : i + h_w + 1] = 1.0
        masks = tf.convert_to_tensor(masks)
        return tf.tile(
            tf.expand_dims(masks, 0),
            [tf.shape(self.X)[0] * self.num_heads, 1, 1],
        )
tf.reset_default_graph()
modelnn = Model(
    df_log.shape[1],
    df_log.shape[1],
    timestamp,
    0.01,
    num_heads = df_log.shape[1],
)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    total_loss = 0
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(
            df_log.iloc[k : k + timestamp].values, axis = 0
        )
        batch_y = df_log.iloc[k + 1 : k + timestamp + 1].values
        _, loss = sess.run(
            [modelnn.optimizer, modelnn.cost],
            feed_dict = {modelnn.X: batch_x, modelnn.Y: batch_y},
        )
        total_loss += loss
    total_loss /= df_log.shape[0] // timestamp
    if (i + 1) % 100 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)

output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
output_predict[0] = df_log.iloc[0]
upper_b = (df_log.shape[0] // timestamp) * timestamp

for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
    out_logits = sess.run(
        modelnn.logits,
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[k : k + timestamp], axis = 0)
        },
    )
    output_predict[k + 1 : k + timestamp + 1] = out_logits

df_log.loc[df_log.shape[0]] = out_logits[-1]
date_ori.append(date_ori[-1] + timedelta(days = 1))

for i in range(future_day - 1):
    out_logits = sess.run(
        modelnn.logits,
        feed_dict = {
            modelnn.X: np.expand_dims(df_log.iloc[-timestamp:], axis = 0)
        },
    )
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

file_path = '21.multihead_{}'.format(file)
result.to_csv(file_path,index=False)