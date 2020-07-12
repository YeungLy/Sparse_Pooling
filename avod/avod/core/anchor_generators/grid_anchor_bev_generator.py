"""
Generates 3D anchors, placing them on the ground plane
"""

import numpy as np

from avod.core import anchor_generator


class GridAnchorBevGenerator(anchor_generator.AnchorGenerator):

    def name_scope(self):
        return 'GridAnchorBevGenerator'

    def _generate(self, **params):
        """
        Generates Bev anchors in a grid in the Bev map and places
        them on the ground_plane.

        Args:
            **params:

        Returns:
            list of 3D anchors in the form N x [x, y, z, l, w, h, ry]
        """

        image_shapes = params.get('image_shapes')
        anchor_base_sizes = params.get('anchor_base_sizes')
        anchor_strides = params.get('anchor_strides')
        anchor_ratios = params.get('anchor_ratios')
        anchor_scales = params.get('anchor_scales')
        init_ry_type = params.get('anchor_init_ry_type')

        init_ry = float(init_ry_type) / 180 * np.pi #degree to rad

        #image shape: (h, w) for each pyramid level
        all_levels_anchors = []
        bev_shape = [700, 800]
        for idx, image_shape in enumerate(image_shapes):
            anchor_stride = anchor_strides[idx]
            anchor_base_size = anchor_base_sizes[idx]
            anchors = generate_anchors(base_size=anchor_base_size, ratios=anchor_ratios, scales=anchor_scales)
            shifted_anchors = shift(image_shape, anchor_stride, anchors)
            too_small = shifted_anchors[:, 3] < 0 
            anchors = corner_to_center(shifted_anchors)
            #anchors = corner_to_center(anchors)
            init_rys = np.zeros((len(anchors), 1)) + init_ry
            #switch w, h to fit ry=pi/2, if ry=0 or pi, then dont switch w,h
            #if init_ry_type == '-90':
            #    anchors[:, [2, 3]] = anchors[:, [3, 2]] 
            anchors = np.hstack([anchors, init_rys])
            all_levels_anchors.append(anchors)
        return all_levels_anchors


def corner_to_center(anchors):
    xmin, ymin, xmax, ymax = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    xc = (xmin + xmax) / 2.
    w = xmax - xmin
    yc = (ymin + ymax) / 2.
    h = ymax - ymin
    return np.stack([xc, yc, w, h], axis=1)

def generate_anchors(base_size, ratios, scales):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """


    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


