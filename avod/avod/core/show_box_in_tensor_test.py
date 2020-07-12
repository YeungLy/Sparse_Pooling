import numpy as np
import cv2

import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import show_box_in_tensor 
from avod.core import constants
from avod.core import box_3d_projector
from avod.core import box_bev_encoder
from avod.core import anchor_bev_encoder
from avod.core.anchor_generators import grid_anchor_bev_generator

def test():
    pipeline_config_path = '/home/amax_yly/Project/sparse_pooling/avod/avod/configs/retinanet_car_SHPL.config' 
    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            pipeline_config_path, is_training=True)
    dataset_config.data_split = 'train_mini'
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)
    sample_indices = [9]
    samples = dataset.load_samples(sample_indices)
    sample = samples[0]
    bev_input = sample.get(constants.KEY_BEV_INPUT)
    bev_input = bev_input[:, :, [-1, 0, 2]]
    anchors_info = sample.get(constants.KEY_ANCHORS_INFO)
    label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)
    _, boxes_bev_pts_norm_gt = box_3d_projector.project_to_bev(label_boxes_3d, dataset.kitti_utils.area_extents[[0, 2]])
    boxes_bev_pts_norm_gt[:, :, 0] *= bev_input.shape[1]  # 3d_x axis * bev_w 
    boxes_bev_pts_norm_gt[:, :, 1] *= bev_input.shape[0]  # 3d_z axis * bev_h 
    boxes_bev_gt = box_bev_encoder.box_bev_4c_to_center(boxes_bev_pts_norm_gt)

    boxes_bev_norm_gt, _, _ = box_3d_projector.project_to_bev_box(label_boxes_3d, dataset.kitti_utils.area_extents[[0, 2]])
    bev_map_h, bev_map_w = bev_input.shape[:2]
    boxes_bev_gt2 = np.multiply(boxes_bev_norm_gt, np.array([bev_map_w, bev_map_h, bev_map_w, bev_map_h, 1]))

    anchor_generator = grid_anchor_bev_generator.GridAnchorBevGenerator()
    anchor_params = dataset.kitti_utils.mini_batch_utils.retinanet_anchor_params
    all_level_anchor_boxes_bev = anchor_generator.generate(\
            image_shapes=anchor_params['image_shapes'],
            anchor_base_sizes=anchor_params['anchor_base_sizes'],
            anchor_strides=anchor_params['anchor_strides'],
            anchor_ratios=anchor_params['anchor_ratios'],
            anchor_scales=anchor_params['anchor_scales'],
            anchor_init_ry_type=anchor_params['anchor_init_ry_type'])
    #concate all levels anchors
    all_anchor_boxes_bev = np.concatenate(all_level_anchor_boxes_bev)
    #print('anchors shape from anchor_generator: ', all_anchor_boxes_bev.sha

    anchor_indices, anchors_ious, anchor_offsets, \
            anchor_classes = anchors_info
    anchor_boxes_bev_to_use = all_anchor_boxes_bev[anchor_indices]

    anchor_boxes_bev_to_use = np.asarray(anchor_boxes_bev_to_use)
    anchors_ious = np.asarray(anchors_ious)
    anchor_offsets = np.asarray(anchor_offsets)
    anchor_offsets_box = anchor_offsets[:, :5]
    pos_anchor_iou = anchors_ious > dataset.kitti_utils.mini_batch_utils.retinanet_pos_iou_range[0]
    demo_offsets = anchor_offsets_box[pos_anchor_iou][:2]
    demo_anchors = anchor_boxes_bev_to_use[pos_anchor_iou][:2]
    print(f'demo anchors: {demo_anchors}\ndemo_offsets:{demo_offsets}')
    pos_anchor_gt = anchor_bev_encoder.offset_to_anchor(demo_anchors, demo_offsets)
    fake_labels = np.ones_like(pos_anchor_gt[:,0], dtype=np.int32) * show_box_in_tensor.CONSTANT_DRAW_BOX_LABEL_TYPE['ONLY_DRAW_BOXES']
    fake_scores = np.zeros_like(pos_anchor_gt[:,0])
    vis_img = show_box_in_tensor.draw_boxes_with_label_and_scores(\
            img_array=bev_input,
            boxes=pos_anchor_gt,
            labels=fake_labels,
            scores=fake_scores,
            method=1)
    cv2.imwrite('vis.png', vis_img)
    vis_gt_img = show_box_in_tensor.draw_boxes_with_label_and_scores(\
            img_array=bev_input,
            boxes=boxes_bev_gt,
            labels=fake_labels,
            scores=fake_scores,
            method=1)
    cv2.imwrite('vis_gt.png', vis_gt_img)
    print('gt projected to bev:\n', boxes_bev_gt)
    print('gt projected to bev box:\n', boxes_bev_gt2)
    print('anchor offsets to gt:\n', pos_anchor_gt)



if __name__ == "__main__":
    test()
