import numpy as np
import cv2
from avod.utils.box_utils.rbbox_overlaps import rbbx_overlaps
from avod.utils.box_utils.rotate_polygon_nms import rotate_gpu_nms

def calculate_rotate_iou(gt_boxes_r, anchors, device_id=0):
    anchors = anchors.astype(np.float32)
    gt_boxes_r = gt_boxes_r.astype(np.float32)
    ious = rbbx_overlaps(np.ascontiguousarray(gt_boxes_r, dtype=np.float32), \
            np.ascontiguousarray(anchors, dtype=np.float32),\
            device_id)
    return ious

def calculate_rotate_iou_cpu(gt_boxes_r, anchors):
    area1 = gt_boxes_r[:, 2] * gt_boxes_r[:, 3]
    area2 = anchors[:, 2] * anchors[:, 3]
    ious = []
    for i, box1 in enumerate(gt_boxes_r):
        temp_ious = []
        temp_inters = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(anchors):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                iou = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(iou)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    ious = np.array(ious, dtype=np.float32)

    return ious
    
def test_calculate_rotate_iou():
    gt_boxes_r = [[1, 1, 1, 1, 30],
            [2, 2, 2, 2, -90],
            ]
    gt_boxes_r = np.array(gt_boxes_r)
    anchors = [[2, 2, 2, 2, 90],
               ]
    anchors = np.array(anchors)
    #anchors[:, 2] = np.ones_like(anchors[:, 2]) * 1e-7
    #anchors[:, 3] = np.ones_like(anchors[:, 3]) * 1e-7
    ious = calculate_rotate_iou(gt_boxes_r, anchors)
    print('gpu result')
    print(ious)
    ious = calculate_rotate_iou_cpu(gt_boxes_r, anchors)
    print('cpu result')
    print(ious)


def test_rotate_nms():
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])

    scores = np.array([0.99, 0.88, 0.66, 0.77])
    det_tensor = np.hstack([boxes, scores[:, np.newaxis]])
    det_tensor = det_tensor.astype(np.float32)
    iou_threshold = 0.4
    print(iou_threshold)
    for i in range(boxes.shape[0]):
        j = (i+1)%(boxes.shape[0])
        iou = calculate_rotate_iou(boxes[i].reshape(1, -1), boxes[j].reshape(1, -1))
        print('gpu result', iou)
        iou = calculate_rotate_iou_cpu(boxes[i].reshape(1, -1), boxes[j].reshape(1, -1))
        print('cpu result', iou)
    device_id = 0
    keep = rotate_gpu_nms(det_tensor, iou_threshold, device_id)
    print(keep)

if __name__ == '__main__':
    #test_calculate_rotate_iou()
    test_rotate_nms()
 
