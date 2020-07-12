import numpy as np
import tensorflow as tf

import avod.core.format_checker as fc

def anchor_to_offset(anchors, gt_anchor, angle_range='180'):
    """Offset between bev anchor and bev gt anchor
    bev anchor format: N x 5 [xc, yc, w, h, angle] 
    """
    fc.check_anchor_bev_format(anchors)

    anchors = np.asarray(anchors).reshape(-1, 5)
    ground_truth = np.reshape(gt_anchor, (5,))

    # t_x_gt = (x_gt - x_anch)/dim_x_anch
    t_x_gt = (ground_truth[0] - anchors[:, 0]) / anchors[:, 2]
    # t_y_gt = (y_gt - y_anch)/dim_y_anch
    t_y_gt = (ground_truth[1] - anchors[:, 1]) / anchors[:, 3]
    # t_dx_gt = log(dim_x_gt/dim_x_anch)
    t_dx_gt = np.log(ground_truth[2] / anchors[:, 2])
    # t_dy_gt = log(dim_y_gt/dim_y_anch)
    t_dy_gt = np.log(ground_truth[3] / anchors[:, 3])
    t_angle_gt = anchor_to_offset_angle(anchors[:, 4], ground_truth[4], angle_range) 
    #t_angle_gt = ground_truth[4] - anchors[:, 4] #in rad format (-pi, pi).
    anchor_offsets = np.stack((t_x_gt,
                               t_y_gt,
                               t_dx_gt,
                               t_dy_gt,
                               t_angle_gt), axis=1)
    return anchor_offsets

def tf_anchor_to_offset(anchors, ground_truth, angle_range='180'):
    """Encodes the anchor regression predictions with the
    ground truth.

    This function assumes the ground_truth tensor has been arranged
    in a way that each corresponding row in ground_truth, is matched
    with that anchor according to the highest IoU.
    For instance, the ground_truth might be a matrix of shape (256, 6)
    of repeated entries for the original ground truth of shape (x, 6),
    where each entry has been selected as the highest IoU match with that
    anchor. This is different from the same function in numpy format, where
    we loop through all the ground truth anchors, and calculate IoUs for
    each and then select the match with the highest IoU.

    Args:
        anchors: A tensor of shape (N, 6) representing
            the generated anchors.
        ground_truth: A tensor of shape (N, 6) containing
            the label boxes in the anchor format. Each ground-truth entry
            has been matched with the anchor in the same entry as having
            the highest IoU.

    Returns:
        anchor_offsets: A tensor of shape (N, 6)
            encoded/normalized with the ground-truth, representing the
            offsets.
    """

    fc.check_anchor_bev_format(anchors)

    # Make sure anchors and anchor_gts have the same shape
    dim_cond = tf.equal(tf.shape(anchors), tf.shape(ground_truth))

    with tf.control_dependencies([dim_cond]):
        t_x_gt = (ground_truth[:, 0] - anchors[:, 0]) / anchors[:, 2]
        t_y_gt = (ground_truth[:, 1] - anchors[:, 1]) / anchors[:, 3]
        t_dx_gt = tf.log(ground_truth[:, 2] / anchors[:, 2])
        t_dy_gt = tf.log(ground_truth[:, 3] / anchors[:, 3])
        #t_angle_gt = ground_truth[:, 4] - anchors[:, 4]
        t_angle_gt = anchor_to_offset_angle(anchors[:, 4], ground_truth[:, 4], angle_range)
        anchor_offsets = tf.stack((t_x_gt,
                                   t_y_gt,
                                   t_dx_gt,
                                   t_dy_gt,
                                   t_angle_gt), axis=1)

        return anchor_offsets


def offset_to_anchor(anchors, offsets):
    """Decodes the anchor regression predictions with the
    anchor.

    Args:
        anchors: A numpy array or a tensor of shape [N, 6]
            representing the generated anchors.
        offsets: A numpy array or a tensor of shape
            [N, 6] containing the predicted offsets in the
            anchor format  [x, y, z, dim_x, dim_y, dim_z].

    Returns:
        anchors: A numpy array of shape [N, 6]
            representing the predicted anchor boxes.
    """

    fc.check_anchor_bev_format(anchors)
    fc.check_anchor_bev_format(offsets)

    # x = dx * dim_x + x_anch
    x_pred = (offsets[:, 0] * anchors[:, 2]) + anchors[:, 0]
    # y = dy * dim_y + x_anch
    y_pred = (offsets[:, 1] * anchors[:, 3]) + anchors[:, 1]
    #angle_pred = offsets[:, 4] + anchors[:, 4]
    angle_pred = offset_to_anchor_angle(anchors[:, 4], offsets[:, 4])

    tensor_format = isinstance(anchors, tf.Tensor)
    if tensor_format:
        # dim_x = exp(log(dim_x) + dx)
        dx_pred = tf.exp(tf.log(anchors[:, 2]) + offsets[:, 2])
        # dim_y = exp(log(dim_y) + dy)
        dy_pred = tf.exp(tf.log(anchors[:, 3]) + offsets[:, 3])
        anchors = tf.stack((x_pred,
                            y_pred,
                            dx_pred,
                            dy_pred,
                            angle_pred), axis=1)
    else:
        dx_pred = np.exp(np.log(anchors[:, 2]) + offsets[:, 2])
        dy_pred = np.exp(np.log(anchors[:, 3]) + offsets[:, 3])
        anchors = np.stack((x_pred,
                            y_pred,
                            dx_pred,
                            dy_pred,
                            angle_pred), axis=1)

    return anchors

def offset_to_anchor_h(anchor_h, offset_h):
    anchor_y3d = anchor_h[:, 0]
    anchor_h3d = anchor_h[:, 1]
    offset_y3d = offset_h[:, 0]
    offset_h3d = offset_h[:, 1]
    tensor_format = isinstance(anchor_h, tf.Tensor)
    if tensor_format:
        pred_y3d = offset_y3d * anchor_h3d + anchor_y3d
        pred_h3d = tf.exp(offset_h3d) * anchor_h3d
        pred_h = tf.stack([pred_y3d, pred_h3d], axis=1)
    else:
        pred_y3d = offset_y3d * anchor_h3d + anchor_y3d
        pred_h3d = np.exp(offset_h3d) * anchor_h3d
        pred_h = np.stack([pred_y3d, pred_h3d], axis=1)

    return pred_h



def anchor_to_offset_h(anchor_h, gt_h):
    anchor_y3d =anchor_h[:, 0]
    anchor_h3d =anchor_h[:, 1]
    tensor_format = isinstance(anchor_h, tf.Tensor)
    if tensor_format:
        gt_y3d = gt_h[:, 0]
        gt_h3d = gt_h[:, 1]
        offset_y3d = (gt_y3d - anchor_y3d) / anchor_h3d
        offset_h3d = tf.log(gt_h3d / anchor_h3d)
        offsets = tf.stack([offset_y3d, offset_h3d], axis=1)
    else:
        gt_y3d = gt_h[0]
        gt_h3d = gt_h[1]
        offset_y3d = (gt_y3d - anchor_y3d) / anchor_h3d
        offset_h3d = np.log(gt_h3d / anchor_h3d)
        offsets = np.stack([offset_y3d, offset_h3d], axis=1)
    return offsets

def get_default_anchor_h(n_anchor, fmt='np'):
    default_y = 1.65
    default_h = 1.53
    if fmt == 'np':
        anchor_y3d = np.ones(n_anchor) * default_y
        anchor_h3d = np.ones(n_anchor) * default_h 
        anchor_h = np.stack([anchor_y3d, anchor_h3d], axis=1)
    elif fmt == 'tf':
        anchor_y3d = tf.ones(n_anchor) * default_y
        anchor_h3d = tf.ones(n_anchor) * default_h
        anchor_h = tf.stack([anchor_y3d, anchor_h3d], axis=1)

    return anchor_h
        


def offset_to_anchor_angle(anchor_angle, offset_angle):
    return anchor_angle + offset_angle

def anchor_to_offset_angle(anchor_angle, gt_angle, angle_range='180'):
    tensor_format = isinstance(anchor_angle, tf.Tensor)
    if angle_range == '180':
        if tensor_format:
            #tf version, gt_angle is (num_anchor, )
            gt_angle_shifted = tf.where(gt_angle < 0,
                                    gt_angle + np.pi,
                                    gt_angle)
        else:
            #np version, gt_angle, just a scalar
            gt_angle_shifted = gt_angle
            if gt_angle < 0:
                gt_angle_shifted += np.pi
        
        offset_angle = gt_angle_shifted - anchor_angle
    else:
        offset_angle = gt_angle - anchor_angle
    return offset_angle


