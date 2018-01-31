import preprocessing as prep
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

rnn_unit = 10
input_size = 3
output_size = 1
lr = 0.006

pred_result = np.zeros(62)
f = 0

datatrain = np.zeros([4, 122])
datatest = np.zeros([4, 62])
init = tf.global_variables_initializer()

for artistNo in range(0,50):
    g = tf.Graph()
    with g.as_default():
        #read data
        datatrain[0] = prep.newsong[artistNo][:122]
        datatrain[1] = prep.collect_train[artistNo]
        datatrain[2] = prep.download_train[artistNo]
        datatrain[3] = prep.play_train[artistNo]
        data_train = datatrain.T

        datatest[0] = prep.newsong[artistNo][122:]
        datatest[1] = prep.collect_test[artistNo]
        datatest[2] = prep.download_test[artistNo]
        datatest[3] = prep.play_test[artistNo]
        data_test = datatest.T


        # training data
        def get_train_data(batch_size, time_step):
            batch_index = []
            train_x, train_y = [], []
            mean = np.mean(data_train, axis=0)
            std = np.std(data_train, axis=0)
            for i in range(0, 4):
                if std[i] == 0:
                    std[i] = 1
            nor_data_train = (data_train - mean) / std
            for i in range(len(nor_data_train) - time_step):
                if i % batch_size == 0:
                    batch_index.append(i)
                x = nor_data_train[i:i + time_step, :3]
                y = nor_data_train[i:i + time_step, 3, np.newaxis]
                train_x.append(x.tolist())
                train_y.append(y.tolist())
            batch_index.append((len(nor_data_train) - time_step))
            return batch_index, train_x, train_y

        # testing data
        def get_test_data(time_step):
            mean = np.mean(data_test, axis=0)
            std = np.std(data_test, axis=0)
            for i in range(0, 4):
                if std[i] == 0:
                    std[i] = 1
            nor_data_test = (data_test - mean) / std
            size = (len(nor_data_test) + time_step - 1) // time_step
            test_x, test_y = [], []
            for i in range(size - 1):
                x = nor_data_test[i * time_step:(i + 1) * time_step, :3]
                y = nor_data_test[i * time_step:(i + 1) * time_step, 3]
                test_x.append(x.tolist())
                test_y.extend(y)
            test_x.append((nor_data_test[(i + 1) * time_step:, :3]).tolist())
            test_y.extend((nor_data_test[(i + 1) * time_step:, 3]).tolist())
            return test_x, test_y, mean, std


        weights = {'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
                   'out': tf.Variable(tf.random_normal([rnn_unit, 1]))}
        biases = {'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
                  'out': tf.Variable(tf.constant(0.1, shape=[1, ]))}

        #build model
        def lstm(X):
            batch_size = tf.shape(X)[0]
            time_step = tf.shape(X)[1]
            w_in = weights['in']
            b_in = biases['in']
            input = tf.reshape(X, [-1, input_size])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, rnn_unit])
            w_out = weights['out']
            b_out = biases['out']
            pred = tf.matmul(output, w_out) + b_out
            return pred, final_states

        #train model
        def train_lstm(batch_size=30, time_step=31):
            X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
            Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
            batch_index, train_x, train_y = get_train_data(batch_size, time_step)
            with tf.variable_scope(str(artistNo)):
                pred, _ = lstm(X)
            loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(30):
                    for step in range(len(batch_index) - 1):
                        _, loss_ = sess.run([train_op, loss],
                                            feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                       Y: train_y[batch_index[step]:batch_index[step + 1]]})
                saver.save(sess, '/Users/zhaozehua/PycharmProjects/music/check/modle' + str(artistNo))

        #predict
        def prediciton_lstm(time_step = 31):
            X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
            test_x, test_y ,mean , std= get_test_data(time_step)
            with tf.variable_scope(str(artistNo),reuse=True):
                pred,_=lstm(X)
            saver = tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                module_file = tf.train.latest_checkpoint("/Users/zhaozehua/PycharmProjects/music/check")
                saver.restore(sess, module_file)
                test_predict = []
                for step in range(len(test_x)):
                    prob = sess.run(pred, feed_dict={X: [test_x[step]]})
                    predict = prob.reshape((-1))
                    test_predict.extend(predict)
                test_predict = np.array(test_predict) * std[3] + mean[3]
                #test_y = np.array(test_y) * std[3] + mean[3]
                return test_predict


        train_lstm()
        pred_result = prediciton_lstm()

        N = 62
        sigma = 0
        phi = 0
        if any(prep.play_test[artistNo]):
            for j in range(0, 62):
                if prep.play_test[artistNo][j] == 0:
                    N -= 1
                else:
                    sigma += ((pred_result[j] - prep.play_test[artistNo][j]) / prep.play_test[artistNo][j]) ** 2
                    phi += prep.play_test[artistNo][j]
        else:
            for j in range(0, 62):
                sigma += ((pred_result[j] - prep.play_test[artistNo][j]) / prep.play_test[artistNo][j]) ** 2
                phi += prep.play_test[artistNo][j]
        s = math.sqrt(sigma / N)
        p = math.sqrt(phi)
        result_f = (1 - s) * p
        f += result_f

        if artistNo < 25:
            plt.figure(1)
            plt.subplot(5, 5, 1 + artistNo)
            plt.plot(list(range(len(pred_result) - 1)), pred_result[:61], color='r', linewidth = 0.5)
            plt.plot(list(range(len(prep.play_test[artistNo]) - 1)), prep.play_test[artistNo][:61], color='b',linewidth = 0.1)
            plt.xticks([])
            plt.yticks([])
            plt.savefig('/Users/zhaozehua/Desktop/ali music/graph-300/1.eps')
        else:
            plt.figure(2)
            plt.subplot(5, 5, 1 + artistNo%25)
            plt.plot(list(range(len(pred_result) - 1)), pred_result[:61], color='r',linewidth = 0.5 )
            plt.plot(list(range(len(prep.play_test[artistNo]) - 1)), prep.play_test[artistNo][:61], color='b',linewidth = 0.1)
            plt.xticks([])
            plt.yticks([])
            plt.savefig('/Users/zhaozehua/Desktop/ali music/graph-300/2.eps')

        print artistNo, result_f

print f

