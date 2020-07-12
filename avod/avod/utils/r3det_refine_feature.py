import tensorflow as tf
from tensorflow.contrib import slim

def refine_feature_op(points, feature_map, output_channel, initializers):

    subnet_weights_initializer = initializers['subnet_weights_initializer']
    subnet_bias_initializer = initializers['subnet_bias_initializer']

    h, w = tf.cast(tf.shape(feature_map)[1], tf.int32), tf.cast(tf.shape(feature_map)[2], tf.int32)

    xmin = tf.maximum(0.0, tf.floor(points[:, 0]))
    ymin = tf.maximum(0.0, tf.floor(points[:, 1]))
    xmax = tf.minimum(tf.cast(w - 1, tf.float32), tf.ceil(points[:, 0]))
    ymax = tf.minimum(tf.cast(h - 1, tf.float32), tf.ceil(points[:, 1]))

    left_top = tf.cast(tf.transpose(tf.stack([xmin, ymin], axis=0)), tf.int32)
    right_bottom = tf.cast(tf.transpose(tf.stack([xmax, ymax], axis=0)), tf.int32)
    left_bottom = tf.cast(tf.transpose(tf.stack([xmin, ymax], axis=0)), tf.int32)
    right_top = tf.cast(tf.transpose(tf.stack([xmax, ymin], axis=0)), tf.int32)

    feature_1x5 = slim.conv2d(inputs=feature_map,
                              num_outputs=output_channel,
                              kernel_size=[1, 5],
                              weights_initializer=subnet_weights_initializer,
                              biases_initializer=subnet_bias_initializer,
                              stride=1,
                              activation_fn=None,
                              scope='refine_1x5')

    feature5x1 = slim.conv2d(inputs=feature_1x5,
                             num_outputs=output_channel,
                             kernel_size=[5, 1],
                             weights_initializer=subnet_weights_initializer,
                             biases_initializer=subnet_bias_initializer,
                             stride=1,
                             activation_fn=None,
                             scope='refine_5x1')

    feature_1x1 = slim.conv2d(inputs=feature_map,
                              num_outputs=output_channel,
                              kernel_size=[1, 1],
                              weights_initializer=subnet_weights_initializer,
                              biases_initializer=subnet_bias_initializer,
                              stride=1,
                              activation_fn=None,
                              scope='refine_1x1')

    feature = feature5x1 + feature_1x1

#feature shape: (N, H, W, C)
#left_top shape: (num_of_pred_boxes, 2) , index at H and W when batchsize=1. 
    left_top_feature = tf.gather_nd(tf.squeeze(feature), left_top)
    right_bottom_feature = tf.gather_nd(tf.squeeze(feature), right_bottom)
    left_bottom_feature = tf.gather_nd(tf.squeeze(feature), left_bottom)
    right_top_feature = tf.gather_nd(tf.squeeze(feature), right_top)

    refine_feature = right_bottom_feature * tf.tile(
        tf.reshape((tf.abs((points[:, 0] - xmin) * (points[:, 1] - ymin))), [-1, 1]),
        [1, output_channel]) \
                     + left_top_feature * tf.tile(
        tf.reshape((tf.abs((xmax - points[:, 0]) * (ymax - points[:, 1]))), [-1, 1]),
        [1, output_channel]) \
                     + right_top_feature * tf.tile(
        tf.reshape((tf.abs((points[:, 0] - xmin) * (ymax - points[:, 1]))), [-1, 1]),
        [1, output_channel]) \
                     + left_bottom_feature * tf.tile(
        tf.reshape((tf.abs((xmax - points[:, 0]) * (points[:, 1] - ymin))), [-1, 1]),
        [1, output_channel])

    refine_feature = tf.reshape(refine_feature, [1, tf.cast(h, tf.int32), tf.cast(w, tf.int32), output_channel])

    # refine_feature = tf.reshape(refine_feature, [1, tf.cast(feature_size[1], tf.int32),
    #                                              tf.cast(feature_size[0], tf.int32), 256])

    return refine_feature + feature


