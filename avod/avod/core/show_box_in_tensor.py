
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2


import tensorflow as tf

CONSTANT_DRAW_BOX_LABEL_TYPE = {
        'ONLY_DRAW_BOXES': -1,
        'ONLY_DRAW_BOXES_WITH_SCORES': -2,
        'NOT_DRAW_BOXES': 0, 
    }
CONSTANT_BOX_COLOR = {
        1:'Red',
        2:'Green',
    }
FONT = ImageFont.load_default()

def only_draw_boxes(img_batch, boxes, method):

    boxes = tf.stop_gradient(boxes)
    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0], ), dtype=tf.int32) * CONSTANT_DRAW_BOX_LABEL_TYPE['ONLY_DRAW_BOXES']
    scores = tf.zeros_like(labels, dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_boxes_with_label_and_scores,
                       inp=[img_tensor, boxes, labels, scores, method],
                       Tout=tf.uint8)
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))  # [batch_size, h, w, c]

    return img_tensor_with_boxes


def draw_boxes_with_scores(img_batch, boxes, scores, method):

    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.int32) * CONSTANT_DRAW_BOX_LABEL_TYPE['ONLY_DRAW_BOXES_WITH_SCORES']
    img_tensor_with_boxes = tf.py_func(draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores, method],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories(img_batch, boxes, labels, method):
    boxes = tf.stop_gradient(boxes)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    scores = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.float32)

    img_tensor_with_boxes = tf.py_func(draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores, method],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores, method):
    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores, method],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_with_label_and_scores(img_array, boxes, labels, scores, method):
    img_array.astype(np.float32)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    num_of_objs = 0
    for box, a_label, a_score in zip(boxes, labels, scores):
        if a_label == CONSTANT_DRAW_BOX_LABEL_TYPE['NOT_DRAW_BOXES']:
            continue
        num_of_objs += 1
        if a_label == CONSTANT_DRAW_BOX_LABEL_TYPE['ONLY_DRAW_BOXES']:
            a_label, a_score = None, None
        elif a_label == CONSTANT_DRAW_BOX_LABEL_TYPE['ONLY_DRAW_BOXES_WITH_SCORES']:
            a_label = None
        color='Red' if a_label is None else CONSTANT_BOX_COLOR[a_label] 
        draw_in_img(draw_obj, box, color=color, width=3, method=method, score=a_score, label=a_label)

    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=0.7)

    return np.array(out_img_obj)

def draw_in_img(draw_obj, box, color, width, method, label=None, score=None):
    '''
    tools for draw_labels_and_scores.
    use draw lines to draw rectangle. since the draw_rectangle func can not modify the width of rectangle
    :param draw_obj:
    :param box: [x1, y1, x2, y2]
    :return:
    '''
    # color = (0, 255, 0)
    if method == 0:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        top_left, top_right = (x1, y1), (x2, y1)
        bottom_left, bottom_right = (x1, y2), (x2, y2)

        draw_obj.line(xy=[top_left, top_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_left, bottom_left],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[bottom_left, bottom_right],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[top_right, bottom_right],
                      fill=color,
                      width=width)
    else:
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        theta *= (180 / np.pi)
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        #print(f'draw_boxes_with_label_and_scores:\nbox:{box},\n8pts:{rect}')
        rect = np.int0(rect)
        draw_obj.line(xy=[(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[1][0], rect[1][1]), (rect[2][0], rect[2][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[2][0], rect[2][1]), (rect[3][0], rect[3][1])],
                      fill=color,
                      width=width)
        draw_obj.line(xy=[(rect[3][0], rect[3][1]), (rect[0][0], rect[0][1])],
                      fill=color,
                      width=width)

    if not label is None or not score is None:
        x, y = box[0], box[1]
        txt_color = 'White'
        draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
                           fill=txt_color)
        label = 'obj' if label is None else label
        txt = '{}:{}'.format(label, round(score,2 ))
        draw_obj.text(xy=(x, y),
                  text=txt,
                  fill='black',
                  font=FONT)

if __name__ == "__main__":
    print (1)

