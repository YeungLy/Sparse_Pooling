import numpy as np
import tensorflow as tf


def tf_orientation_to_angle_vector(orientations_tensor):
    """ Converts orientation angles into angle unit vector representation.
        e.g. 45 -> [0.717, 0.717], 90 -> [0, 1]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 2) of angle unit vectors in the format [x, y]
    """
    x = tf.cos(orientations_tensor)
    y = tf.sin(orientations_tensor)

    return tf.stack([x, y], axis=1)


def tf_angle_vector_to_orientation(angle_vectors_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.717, 0.717] -> 45, [0, 1] -> 90

    Args:
        angle_vectors_tensor: a tensor of shape (N, 2) of angle unit vectors
            in the format [x, y]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    x = angle_vectors_tensor[:, 0]
    y = angle_vectors_tensor[:, 1]

    return tf.atan2(y, x)


def orientation_to_angle_cls(orientations_array):
    angle_cls = np.ones(orientations_array.shape[0], np.int32)
    angle_cls[orientations_array >= 0] = 0
    #n_cls = 2
    #one-hot encode
    #angle_cls_logits = np.eye(n_cls)[angle_cls].T
    #return angle_cls_logits
    return angle_cls
    

def tf_orientation_to_angle_cls(orientations_tensor):
    angle_cls = tf.where(orientations_tensor >= 0, 
            tf.zeros_like(orientations_tensor), 
            tf.ones_like(orientations_tensor))
    angle_cls_logits = tf.one_hot(
            tf.cast(angle_cls, tf.int32),
            depth=2,
            )
    return angle_cls_logits


def angle_clsval_to_orientation(angle_cls, angle_val):
#For evaluation
    #angle_cls = np.argmax(angle_cls_logits, axis=-1)
    angle_cls = angle_cls.astype(np.float32)
    ori_shifted = angle_val
    #make sure ori_shidted is in [0, pi], then subtract pi by angle_cls 
    ori = np.arccos(np.cos(ori_shifted)) - angle_cls * np.pi
    too_much = np.logical_or(ori > np.pi, ori < -np.pi)

    return ori
    
def tf_angle_clsval_to_orientation(angle_cls, angle_val):
#For evaluation
    #angle_cls = tf.argmax(angle_cls_logits, axis=-1)
    angle_cls = tf.cast(angle_cls, tf.float32)
    ori_shifted = angle_val
    #make sure ori_shidted is in [0, pi]
    ori = tf.acos(tf.cos(ori_shifted)) - angle_cls * np.pi
    return ori
    
