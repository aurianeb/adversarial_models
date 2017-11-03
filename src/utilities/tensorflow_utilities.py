# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import tensorflow as tf


def fc_layer(x, size_in, size_out, activation=tf.nn.relu, name="fc"):
    """
    Implements a fully connected layer in TensorFlow

    Args:
        x ():
        size_in ():
        size_out ():
        activation ():
        name ():

    Returns:
        activation Tensor

    """
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        z = tf.add(tf.matmul(x, w), b, name='z')
        if activation == 'linear':
            a = z
        else:
            a = activation(z)

        # TensorBoard
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("z", z)
        tf.summary.histogram("activation", a)
        return a


def conv_layer(x, size_in, size_out, name="conv"):
    """
    Implements a convnet layer in TensorFlow
    Args:
        x ():
        size_in ():
        size_out ():
        name ():

    Returns:
        activation Tensor

    """
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)

        # TensorBoard
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def leaky_relu(x, leakiness=0.1):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def causal_convolution_layer(input_seq, width, n_dims_out,
                             dilation=1, causal=True,
                             non_linearity=leaky_relu):
    """
    Causal convolution for prediction.
    Args:
        non_linearity ():
        input_seq (tf tensor): shape (batch_size, length, n_dims_in).
        width (int):
        n_dims_out (int):
        dilation ():
        causal:

    Returns:
        tensor with shape (batch_size, length, n_dims_out).
    """

    n_dims_in = input_seq.get_shape().as_list()[-1]

    conv_kernel = tf.get_variable(
        name="kernel",
        shape=[width, n_dims_in, n_dims_out],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer())

    # Similar to approach in Francois Chollet's Keras library
    if causal:
        offset = dilation * (width - 1)
        input_seq = tf.pad(input_seq, [[0, 0], [offset, 0], [0, 0]])

    conv_output = tf.nn.convolution(
        input=input_seq,
        filter=conv_kernel,
        padding="VALID" if causal else "SAME",
        strides=None,
        dilation_rate=[dilation]
    )

    bias = tf.get_variable(
        name="bias",
        shape=[n_dims_out],
        dtype=tf.float32,
        initializer=tf.zeros_initializer()
    )

    return non_linearity(tf.nn.bias_add(conv_output, bias))


def causal_conv_net(input_seq, kernel_specs,
                    causal=True, non_linearity=leaky_relu):
    """
    Pipeline causal convolution layers.
    """

    output_seq = input_seq

    for i, (dilation, width, n_dims_out) in enumerate(kernel_specs):
        with tf.variable_scope("conv_layer_%d" % i):
            output_seq = causal_convolution_layer(input_seq=output_seq,
                                                  width=width,
                                                  n_dims_out=n_dims_out,
                                                  dilation=dilation,
                                                  causal=causal,
                                                  non_linearity=non_linearity)

    return output_seq
