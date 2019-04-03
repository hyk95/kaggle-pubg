import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib import layers
import functools
from sklearn.model_selection import KFold
from DataReader import load_data


class Tfmodel():
    def __init__(self):
        self.sess = None
        self.inputs = None
        self.y_true = None
        self.y_pred = None

    def create_net(self, inputs, training):
        # initializer = tf.truncated_normal_initializer(stddev=0.001)
        activation_fun = functools.partial(tf.nn.leaky_relu, alpha=0.1)
        initializer = layers.variance_scaling_initializer()
        kernel_regularizer = None
        net = tf.layers.dense(inputs, units=128, activation=activation_fun, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)
        conn = net
        net = tf.layers.dense(net, units=64, activation=activation_fun, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)
        net = tf.layers.dense(net, units=64, activation=activation_fun, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)
        net = tf.layers.dense(net, units=128, activation=activation_fun, kernel_initializer=initializer, kernel_regularizer=kernel_regularizer)
        net = tf.add(conn, net)
        net = tf.layers.dense(net, units=128, activation=activation_fun, kernel_initializer=initializer,
                              kernel_regularizer=kernel_regularizer)
        net = tf.layers.dense(net, units=1, activation=None, kernel_initializer=initializer, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0000001))
        return net

    def loss_function(self, y_true, y_pred):
        tf.losses.mean_squared_error(y_true, y_pred)
        weight_loss = tf.losses.get_regularization_losses()
        net_loss = tf.losses.get_losses()
        cost = tf.add_n(weight_loss) + net_loss[0]
        return cost

    def train(self, train_x, train_y, batch_size, epochs):
        num_data = len(train_x)
        self.inputs = tf.placeholder(tf.float32, [None, train_x.shape[1]])
        self.y_true = tf.placeholder(tf.float32, [None, 1])
        self.y_pred = self.create_net(self.inputs, training=True)
        model_loss = self.loss_function(self.y_true, self.y_pred)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=num_data / batch_size,
                                                   decay_rate=0.98, staircase=True)
        with tf.name_scope('evaluation'):
            evaluation_step = tf.reduce_mean(tf.cast(tf.abs(self.y_true-self.y_pred), tf.float32))
        with tf.control_dependencies(update_ops):
            _optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optim = _optim.minimize(model_loss, global_step=global_step)
        self.sess = tf.Session()
        print("init train")
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            startTime = time.time()
            for iter_ in range(num_data // batch_size):
                start = iter_ * batch_size
                end = (iter_ + 1) * batch_size
                x = train_x[start:end]
                y = train_y[start:end]
                _ = self.sess.run([optim], feed_dict={self.inputs: x, self.y_true: y})
                if iter_ % 20000 == 0:
                    loss, score, step, lr = self.sess.run([model_loss, evaluation_step, global_step, learning_rate], feed_dict={self.inputs: x, self.y_true: y})
                    print("epoch:{};iter:{};train_loss:{};score:{},step:{};lr:{}".format(epoch, iter_, loss, score, step, lr))
            endTime = time.time()
            print("epoch_time:{}".format(endTime - startTime))

    def eval(self, test_x, batch_size=1024):
        num_data = len(test_x)
        y_test = np.zeros([num_data, 1])
        for iter_ in range(num_data // batch_size):
            start = iter_ * batch_size
            end = (iter_ + 1) * batch_size
            x = test_x[start:end]
            res = self.sess.run(self.y_pred, feed_dict={self.inputs: x, self.y_true: np.zeros([x.shape[0], 1])})
            y_test[start: end] = res
        return y_test

    def closeSession(self):
        self.sess.close()


if __name__ == '__main__':
    train_X, train_y, test_X = load_data()
    train_meta = np.zeros(train_y.shape[0])
    test_meta = np.zeros(test_X.shape[0])
    splits = list(KFold(n_splits=5, shuffle=True, random_state=233).split(train_X, train_y))
    for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]
        model = Tfmodel()
        model.train(X_train, y_train, batch_size=1280, epochs=250)
        pred_val_y = model.eval(X_val)
        pred_test_y = model.eval(test_X)
        model.closeSession()
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)
    np.save("../mergenpy/nnmodel_train_pred.npy", train_meta)
    np.save("../mergenpy/nnmodel_test_pred.npy", test_meta)

