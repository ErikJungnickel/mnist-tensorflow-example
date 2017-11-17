from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# this is basically just the example from the tensorflow getting started guide


def main(_):
    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y), reduction_indices=[1]))

    # gradient descent adjusting weights based on the cross entropy loss function
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # train
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test set %f' % sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
