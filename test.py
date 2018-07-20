import tensorflow as tf
import numpy as np


def recognize( pb_file_path):
    cat_data = np.load('/home/zhouge/桌面/image/cat.npy')
    circle_data = np.load('/home/zhouge/桌面/image/circle.npy')
    fish_data = np.load('/home/zhouge/桌面/image/fish.npy')
    flower_data = np.load('/home/zhouge/桌面/image/flower.npy')

    train_flag = []
    for i in range(110000):
        train_flag.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        train_flag.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        train_flag.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        train_flag.append(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))

    train_imageData = []
    for i in range(110000):
        train_imageData.append(cat_data[i])
        train_imageData.append(circle_data[i])
        train_imageData.append(fish_data[i])
        train_imageData.append(flower_data[i])

    test_flag = []
    for i in range(1000):
        test_flag.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        test_flag.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        test_flag.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        test_flag.append(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))

    test_imageData = []
    for i in range(1000):
        test_imageData.append(cat_data[i + 110000])
        test_imageData.append(circle_data[i + 110000])
        test_imageData.append(fish_data[i + 110000])
        test_imageData.append(flower_data[i + 110000])

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            print(input_x)
            keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
            print(keep_prob)
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            print(out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            print(out_label)


            img_out_softmax = sess.run(out_softmax, feed_dict={input_x: np.reshape(fish_data[11009], [-1, 784]), keep_prob: 1.0})

            print("img_out_softmax:", img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print("label:", prediction_labels)


recognize( "lenet5.pb")