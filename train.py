import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util



cat_data=np.load('cat.npy')
circle_data=np.load('circle.npy')
fish_data=np.load('fish.npy')
flower_data=np.load('flower.npy')

train_flag =[]
for i in range(110000):
    train_flag.append(np.array([1, 0, 0, 0]))
    train_flag.append(np.array([0, 1, 0, 0]))
    train_flag.append(np.array([0, 0, 1, 0]))
    train_flag.append(np.array([0, 0, 0, 1]))


train_imageData=[]
for i in range(110000):
    train_imageData.append(cat_data[i])
    train_imageData.append(circle_data[i])
    train_imageData.append(fish_data[i])
    train_imageData.append(flower_data[i])

test_flag =[]
for i in range(1000):
    test_flag.append(np.array([1, 0, 0, 0]))
    test_flag.append(np.array([0, 1, 0, 0]))
    test_flag.append(np.array([0, 0, 1, 0]))
    test_flag.append(np.array([0, 0, 0, 1]))


test_imageData=[]
for i in range(1000):
    test_imageData.append(cat_data[i+110000])
    test_imageData.append(circle_data[i+110000])
    test_imageData.append(fish_data[i+110000])
    test_imageData.append(flower_data[i+110000])


def build_network():
    x = tf.placeholder("float", shape=[None, 784], name='input')
    y = tf.placeholder("float", shape=[None, 4], name='labels')
    keep_prob = tf.placeholder("float", name='keep_prob')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # convolution layer
    def lenet5_layer(layer, weight, bias):
        W_conv = weight_variable(weight)
        b_conv = bias_variable(bias)
        h_conv = conv2d(layer, W_conv) + b_conv
        return max_pool_2x2(h_conv)

    # connected layer
    def dense_layer(layer, weight, bias):
        W_fc = weight_variable(weight)
        b_fc = bias_variable(bias)
        return tf.matmul(layer, W_fc) + b_fc

    # first layer
    with tf.name_scope('first') as scope:
        x_image = tf.pad(tf.reshape(x, [-1, 28, 28, 1]), [[0, 0], [2, 2], [2, 2], [0, 0]])
        firstlayer = lenet5_layer(x_image, [5, 5, 1, 6], [6])

    # second layer
    with tf.name_scope('second') as scope:
        secondlayer = lenet5_layer(firstlayer, [5, 5, 6, 16], [16])

    # third layer
    with tf.name_scope('third') as scope:
        W_conv3 = weight_variable([5, 5, 16, 120])
        b_conv3 = bias_variable([120])
        thirdlayerconv = conv2d(secondlayer, W_conv3) + b_conv3
        thirdlayer = tf.reshape(thirdlayerconv, [-1, 120])

    # dense layer1
    with tf.name_scope('dense1') as scope:
        dense_layer1 = dense_layer(thirdlayer, [120, 84], [84])

    # dense layer2
    with tf.name_scope('dense2') as scope:
        dense_layer2 = dense_layer(dense_layer1, [84, 4], [4])

    finaloutput = tf.nn.softmax(tf.nn.dropout(dense_layer2, keep_prob), name="softmax")
    prediction_labels = tf.argmax(finaloutput, axis=1, name="output")

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finaloutput))
    optimize = tf.train.AdamOptimizer(1e-4).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(finaloutput, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return dict(
        x=x,
        y=y,
        keep_prob=keep_prob,
        optimize=optimize,
        cost=cost,
        correct_prediction=correct_prediction,
        accuracy=accuracy,
    )


def train_network(graph, pb_file_path):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        count=0
        for i in range(7000):
            batch = (train_imageData[count:count + 49], train_flag[count:count + 49])
            count+=50
            if i % 100 == 0:
                train_accuracy = sess.run([graph['accuracy']], feed_dict={graph['x']: batch[0], graph['y']: batch[1],
                                                                          graph['keep_prob']: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy[0]))
            sess.run([graph['optimize']],
                     feed_dict={graph['x']: batch[0], graph['y']: batch[1], graph['keep_prob']: 0.5})

        test_accuracy = sess.run([graph['accuracy']],
                                 feed_dict={graph['x']: test_imageData, graph['y']: test_flag,
                                            graph['keep_prob']: 1.0});
        print("Test accuracy %g" % test_accuracy[0])

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

def main():
    pb_file_path = "lenet5.pb"
    g = build_network()
    train_network(g, pb_file_path)


main()