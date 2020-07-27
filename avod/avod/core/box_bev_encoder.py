import numpy as np
import cv2
from avod.core import box_3d_projector
from avod.core import anchor_bev_encoder

def box_bev_to_iou_h_format(boxes_bev):
    """Return min area aligned box cover the rotate bev box
    Args:
        box_bev: BEV boxes in the format [xc, yc, w, h, angle], at BEV map coordinate.
                 shape: (N, 7)
    Returns:
        min area aligned box: [xmin, ymin, xmax, ymax]
                 shape: (N. 4)
    """
    boxes_iou_h_format = []
    for box_bev in boxes_bev:
        xc, yc, w, h, ry = box_bev
        angle = ry / np.pi * 180
        box_bev_pts = cv2.boxPoints(((xc, yc), (w, h), angle))  #shape: (4,2)
        xmin = np.min(box_bev_pts[:, 0])
        ymin = np.min(box_bev_pts[:, 1])
        xmax = np.max(box_bev_pts[:, 0])
        ymax = np.max(box_bev_pts[:, 1])
        iou_h_format = np.array([xmin, ymin, xmax, ymax])
        boxes_iou_h_format.append(iou_h_format)
    boxes_iou_h_format = np.asarray(boxes_iou_h_format, dtype=np.float32)
    return boxes_iou_h_format

def box_bev_to_box_3d(boxes_bev, bev_shape, bev_extents, h3d=None):
    #set default value
    bev_h, bev_w = bev_shape
    #divide bev map shape to normalize box coordinate.
    input_boxes_center_norm = boxes_bev / [bev_w, bev_h, bev_w, bev_h, 1.0]   
    x_extents_min = bev_extents[0][0]
    z_extents_min = bev_extents[1][1]  # z axis is reversed
    x_extents_range = bev_extents[0][1] - bev_extents[0][0]
    z_extents_range = bev_extents[1][0] - bev_extents[1][1]
    #input_boxes_center_norm: (xc, yc, w, h, angle), relative coordinate, 
    boxes_shifted = input_boxes_center_norm * [x_extents_range, z_extents_range, 1.0, 1.0, 1.0]
    boxes = boxes_shifted + [x_extents_min, z_extents_min, 0.0, 0.0, 0.0]
    #[x, y, z, h, w ,l, ry]
    x, z, l_norm, w_norm, ry = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    w = w_norm / np.sqrt((np.sin(ry)**2/x_extents_range**2) + (np.cos(ry)**2/z_extents_range**2)) 
    l = l_norm / np.sqrt((np.cos(ry)**2/x_extents_range**2) + (np.sin(ry)**2/z_extents_range**2)) 
    if not h3d is None:
        y = h3d[:, 0]
        h = h3d[:, 1]
    else:
        h3d = anchor_bev_encoder.get_default_anchor_h(n_anchor=x.shape[0])
        y = h3d[:, 0]
        h = h3d[:, 1]
    return np.stack([x, y, z, l, w, h, ry], axis=1).astype(np.float32)



def box_bev_to_anchor_3d(boxes_bev, bev_shape, bev_extents, h3d=None):
#set default value 
    boxes_3d = box_bev_to_box_3d(boxes_bev, bev_shape, bev_extents, h3d)
    anchors_3d = boxes_3d[:, :-1]
    #(x, y, z, l, w, h, ry) to (x, y, z, dx=l, dy=h, dz=w) 
    anchors_3d[:, [4, 5]] = anchors_3d[:, [5, 4]] 
    return anchors_3d

def box_bev_4c_to_center(boxes_bev):
    boxes_bev_center = []
    for box in boxes_bev:
        pts = box.reshape((4, 2))
        pts = pts.astype(np.int32)
        bx = cv2.minAreaRect(pts)
        xc, yc, w, h, angle = bx[0][0], bx[0][1], bx[1][0], bx[1][1], bx[2]
        angle *= (np.pi/180)
        bx = np.array([xc, yc, w, h, angle])
        boxes_bev_center.append(bx)
    boxes_bev_center = np.asarray(boxes_bev_center)
    return boxes_bev_center

def boxes_3d_project_to_image(boxes_3d, stereo_calib_p2):
    boxes_2d = []
    for box in boxes_3d:
        box2d = box_3d_projector.project_to_image_space(box, stereo_calib_p2)
        boxes_2d.append(box2d)
    boxes_2d = np.asarray(boxes_2d, dtype=np.float32)
    return boxes_2d
    
        

