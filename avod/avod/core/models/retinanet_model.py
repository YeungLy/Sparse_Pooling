import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from avod.builders import feature_extractor_builder
from avod.core import anchor_bev_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import box_3d_encoder
from avod.core import box_bev_encoder
from avod.core import orientation_encoder
from avod.core import constants
from avod.core import losses
from avod.core import model
from avod.core import summary_utils
from avod.core.anchor_generators import grid_anchor_bev_generator
from avod.datasets.kitti import kitti_aug
from avod.utils.sparse_pool_utils import sparse_pool_layer
from avod.core.feature_extractors.fusion_vgg_pyramid import FusionVggPyr
from avod.core import show_box_in_tensor
from avod.core import box_3d_projector
from avod.utils.box_utils.rotate_polygon_nms import rotate_gpu_nms
from avod.utils.r3det_refine_feature import refine_feature_op
from avod.core import box_list
from avod.core import box_list_ops
from avod.core import rotate_iou


class RetinanetModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_ANCHORS = 'anchors_pl'

    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    PL_ANCHOR_IOUS = 'anchor_ious_pl'
    PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    PL_ANCHOR_OFFSETS_H = 'anchor_offsets_h_pl'
    PL_ANCHOR_OFFSETS_ANGLE_CLS = 'anchor_offsets_angle_cls_pl'
    PL_ANCHOR_CLASSES = 'anchor_classes_pl'
    PL_ANCHOR_INDICES = 'anchor_indices_pl'
    PL_ANCHOR_MASK = 'anchor_mask_pl'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    #WZN: placeholder for sparse_pooling_inputs
    PL_M_VAL = 'matrix_f2b_value'
    PL_M_IJ = 'matrix_f2b_value_indices'
    PL_M_SIZE = 'matrix_f2b_size'
    PL_IMG_POOL_IJ = 'image_pool_indices'
    PL_BEV_POOL_IJ = 'bev_pool_indices'

    PL_M_VAL_VGG = 'matrix_f2b_value_after_vgg'
    PL_M_IJ_VGG = 'matrix_f2b_value_indices_after_vgg'
    PL_M_SIZE_VGG = 'matrix_f2b_size_after_vgg'
    PL_IMG_POOL_IJ_VGG = 'image_pool_indices_after_vgg'
    PL_BEV_POOL_IJ_VGG = 'bev_pool_indices_after_vgg'
    
    NET_CLS_SCORES_LIST = 'cls_scores_list'
    NET_CLS_PROBS_LIST = 'cls_probs_list'
    NET_REG_BOXES_LIST = 'reg_boxes_list'
    NET_REG_H_LIST = 'reg_h_list'
    NET_REG_ANGLE_CLS_LIST = 'reg_angle_cls_list'
    NET_ANCHOR_INDICES = 'non_empty_anchor_indices'


    ##############################
    # Keys for Predictions
    ##############################
    STAGE_KEYS = ['fcn']
    PRED_ANCHORS = 'retinanet_anchors'
    PRED_BOXES_LIST = 'retinanet_boxes_list'

    PRED_OBJECTNESS_GT = 'retinanet_objectness_gt'
    PRED_OFFSETS_GT = 'retinanet_offsets_gt'
    PRED_OFFSETS_H_GT = 'retinanet_offsets_h_gt'
    PRED_OFFSETS_ANGLE_CLS_GT = 'retinanet_offsets_angle_cls_gt'

    #PRED_MASK = 'retinanet_mask'
    PRED_OBJECTNESS = 'retinanet_objectness'
    PRED_OFFSETS = 'retinanet_offsets'
    PRED_OFFSETS_H = 'retinanet_offsets_h'
    PRED_OFFSETS_ANGLE_CLS = 'retinanet_offsets_angle_cls'

    PRED_TOP_INDICES = 'retinanet_top_indices'
    PRED_TOP_ANCHORS = 'retinanet_top_anchors'
    PRED_TOP_OBJECTNESS_SIGMOID = 'retinanet_top_objectness_sigmoid'

    #for computing loss.
    POS_ANCHORS_MASK = 'pos_anchors_mask' 
    NEG_ANCHORS_MASK = 'neg_anchors_mask' 
    ##############################
    # Keys for Loss
    ##############################
    LOSS_RETINANET_OBJECTNESS = 'retinanet_objectness_loss'
    LOSS_RETINANET_REGRESSION = 'retinanet_regression_loss'
    LOSS_RETINANET_H = 'retinanet_h_loss'
    LOSS_RETINANET_ANGLE_CLS = 'retinanet_angle_cls_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(RetinanetModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth


        retinanet_config = self._config.retinanet_config
        # Retinanet config
        self._use_sparse_pooling = retinanet_config.use_sparse_pooling and dataset.output_indices
        if self._use_sparse_pooling:
            self._use_pyramid_level_at_SHPL = retinanet_config.use_pyramid_level_at_SHPL
            print('WZN: Using Non-homogeneous Sparse Pooling Layer. \n')

        self._nms_size = retinanet_config.nms_size
        self._nms_iou_thresh = retinanet_config.nms_iou_thresh
        self.refine_stage_num = retinanet_config.refine_stage_num
        self.STAGE_KEYS += [f'refine{i}' for i in range(self.refine_stage_num)]
        self.add_h = retinanet_config.add_h
        self.add_angle = retinanet_config.add_angle
        self.add_h_flags = [False] * (1 + self.refine_stage_num)
        self.add_angle_flags = [False] * (1 + self.refine_stage_num)
        self.add_h_flags[-1] = self.add_h 
        self.add_angle_flags[-1] = self.add_angle 
        self.do_nms_at_gpu = True
        self.reg_target_scales = []

        # Feature Extractor Nets
        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.bev_feature_extractor)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)

        self._feature_pyramid_levels = self._bev_feature_extractor.config.pyramid_levels
        self._fpn_channel = self._bev_feature_extractor.FPN_CHANNEL

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_params = self.dataset.kitti_utils.mini_batch_utils.retinanet_anchor_params
        self._num_anchors_per_location = len(self._anchor_params['anchor_scales']) * \
                                        len(self._anchor_params['anchor_ratios'])

        self._anchor_generator = \
            grid_anchor_bev_generator.GridAnchorBevGenerator()
        self._num_locations_per_level = [shape[0]*shape[1]\
                for shape in self._anchor_params['image_shapes']]
        self._num_anchors_per_level = [loc*self._num_anchors_per_location\
                for loc in self._num_locations_per_level]

        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples


    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)

            self._bev_input_batches = tf.expand_dims(
                bev_input_placeholder, axis=0)

            self._bev_preprocessed = \
                self._bev_feature_extractor.preprocess_input(
                    self._bev_input_batches, self._bev_pixel_size)

            # Summary Images
            bev_summary_images = tf.split(
                bev_input_placeholder, self._bev_depth, axis=2)
            #tf.summary.image("bev_maps", bev_summary_images,
            tf.summary.image("bev_maps", self._bev_preprocessed,
                             max_outputs=2)
                             #max_outputs=self._bev_depth)

        with tf.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(
                tf.float32,
                [None, None, self._img_depth],
                self.PL_IMG_INPUT)

            self._img_input_batches = tf.expand_dims(
                img_input_placeholder, axis=0)

            self._img_preprocessed = \
                self._img_feature_extractor.preprocess_input(
                    self._img_input_batches, self._img_pixel_size)

            # Summary Image
            tf.summary.image("rgb_image", self._img_preprocessed,
                             max_outputs=2)

        #WZN: define sparse pooling inputs
        with tf.variable_scope('pl_sparse_pooling'):
            self._add_placeholder(tf.int64, [None, 2],
                                  self.PL_M_IJ)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_M_VAL)
            self._add_placeholder(tf.int64, [2],
                                  self.PL_M_SIZE)
            self._add_placeholder(tf.int32, [None, 3],
                                  self.PL_IMG_POOL_IJ)
            self._add_placeholder(tf.int32, [None, 3],
                                  self.PL_BEV_POOL_IJ)
        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 7],
                                      self.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.float32, [None],
                                      self.PL_LABEL_CLASSES)

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 5],
                                  self.PL_ANCHORS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_IOUS)
            self._add_placeholder(tf.float32, [None, 5],
                                  self.PL_ANCHOR_OFFSETS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_CLASSES)
            self._add_placeholder(tf.float32, [None, 2],
                                  self.PL_ANCHOR_OFFSETS_H)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_OFFSETS_ANGLE_CLS)
            self._add_placeholder(tf.int32, [None],
                                  self.PL_ANCHOR_INDICES)
            self._add_placeholder(tf.bool, [None],
                                  self.PL_ANCHOR_MASK)

        with tf.variable_scope('sample_info'):
            # the calib matrix shape is (3 x 4)
            self._add_placeholder(
                tf.float32, [3, 4], self.PL_CALIB_P2)
            self._add_placeholder(tf.int32,
                                  shape=[1],
                                  name=self.PL_IMG_IDX)
            self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and
        bottlenecks as member variables.
        """
        # Note: Even if we use the fusion feature extractor, we still need to rely on the pre-processing of infividual 
        self.bev_feature_pyramids = \
            self._bev_feature_extractor.build(
                self._bev_preprocessed,
                self._is_training)
        print('bev feature pyramid keys: ', self.bev_feature_pyramids.keys())


        self.img_feature_pyramids = \
            self._img_feature_extractor.build(
                self._img_preprocessed,
                self._is_training)
        print('img feature pyramid keys: ', self.img_feature_pyramids.keys())

        #WZN: add a sparse pooling based fusion procudure to see the performance change
        if self._use_sparse_pooling:
            #raise NotImplementedError('Not implementation for sparse_pool_layer at fusion bev and img after resnet-fpn.')
            use_level = self._use_pyramid_level_at_SHPL
            if use_level not in self.bev_feature_pyramids.keys() or \
                    use_level not in self.img_feature_pyramids.keys():
                raise IndexError('Invalid pyramid level {} to do SHPL fusion.'.format(use_level))
            bev_feature_maps = self.bev_feature_pyramids[use_level]
            img_feature_maps = self.img_feature_pyramids[use_level]

            #just incase variable dont define.
            self.M_tf = tf.SparseTensor(indices=self.placeholders[self.PL_M_IJ],\
                           values=self.placeholders[self.PL_M_VAL],\
                           dense_shape=self.placeholders[self.PL_M_SIZE])
            #feature depths at all fusion levels at resnet-fpn
            feature_depths = [self._img_feature_extractor.FPN_CHANNEL, self._bev_feature_extractor.FPN_CHANNEL]
            #WZN: the depth of the output feature map (for pooled features bev, img respectively)
            #only fusion on the bev_map
            bev_feature_maps, img_feature_maps = sparse_pool_layer(\
                    [bev_feature_maps,img_feature_maps],feature_depths, self.M_tf, 
                    img_index_flip = self.placeholders[self.PL_IMG_POOL_IJ], \
                    bv_index = None, training=self._is_training)
            print('WZN: Successfully created SHPL \n')
            #do concat at sparse_pool_layer
            #use conv to compress feature depths
            bev_feature_maps = slim.conv2d(
                    bev_feature_maps,
                    feature_depths[0],
                    [3, 3],
                    scope='pyramid_fusion_pooled_bev')

            #update fused feature maps
            self.bev_feature_pyramids[use_level] = bev_feature_maps
            #img_feature_maps = slim.conv2d(
            #        img_feature_maps,
            #        feature_depths[1],
            #        [3, 3],
            #        scope='pyramid_fusion_pooled_img')
            #print('WZN: Sucessfully created conv after fusion')

            #self.img_feature_pyramids[use_level] = img_feature_maps

            self.bev_end_points = self.bev_feature_pyramids
            self.img_end_points = self.img_feature_pyramids
            vis_layer = [use_level, 'P4']
            with tf.variable_scope('bev_feature_maps'):
                #for layer_name in self.bev_end_points:
                for layer_name in vis_layer:
                    summary_utils.add_feature_maps_from_dict(self.bev_end_points,
                                                             layer_name)
            with tf.variable_scope('img_feature_maps'):
                for layer_name in vis_layer:
                    summary_utils.add_feature_maps_from_dict(self.img_end_points,
                                                         layer_name)



        else:
            print('WZN: Not using sparse pooling right before fpn')

    def _fcn_cls_net(self, inputs, scope_postfix, initializers, num_anchors_per_location):
        subnet_weights_initializer = initializers['subnet_weights_initializer']
        subnet_bias_initializer = initializers['subnet_bias_initializer']
        final_conv_bias_initializer = initializers['final_conv_bias_initializer']
        with tf.variable_scope('cls_conv2d'+scope_postfix):
            fcn_cls_conv2d = inputs
            activation_fn=tf.nn.relu
            #activation_fn=tf.nn.leaky_relu
            for i in range(4):
                fcn_cls_conv2d = slim.conv2d(inputs=fcn_cls_conv2d,
                        num_outputs=self._fpn_channel,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=activation_fn,
                        weights_initializer=subnet_weights_initializer,
                        biases_initializer=subnet_bias_initializer,
                        scope='sub{}'.format(i),)
                tf.summary.histogram('histogram_'+fcn_cls_conv2d.name.replace(':', '_'),
                    fcn_cls_conv2d)
        with tf.variable_scope('output'+scope_postfix):
            fcn_cls_scores = slim.conv2d(fcn_cls_conv2d, 
                    num_outputs=self.dataset.num_classes * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1,
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=final_conv_bias_initializer,
                    activation_fn=None,)
            tf.summary.histogram('histogram_'+fcn_cls_scores.name.replace(':', '_'),
                fcn_cls_scores)
            #output shape is (feat_h*feat_w*num_anchors_per_location, num_classes]
            fcn_cls_scores = tf.reshape(fcn_cls_scores, [-1, self.dataset.num_classes],
                    name='reshape')
            fcn_cls_probs = tf.sigmoid(fcn_cls_scores, name='sigmoid')
        return fcn_cls_scores, fcn_cls_probs

    def _fcn_reg_net(self, inputs, scope_postfix, initializers, num_anchors_per_location, add_h, add_angle, share_version=1):
        if share_version == 1:
            return self._fcn_reg_net_sharedv1(inputs, scope_postfix, initializers, num_anchors_per_location,\
                    add_h, add_angle)
        elif share_version == 2:
            return self._fcn_reg_net_sharedv2(inputs, scope_postfix, initializers, num_anchors_per_location,\
                    add_h, add_angle)
        else:
            raise ValueError('Wrong shared version at fcn_reg_net: {}, should be [1, 2]'.format(share_version))

    def _fcn_reg_net_sharedv2(self, inputs, scope_postfix, initializers, num_anchors_per_location, add_h, add_angle):
        subnet_weights_initializer = initializers['subnet_weights_initializer']
        subnet_bias_initializer = initializers['subnet_bias_initializer']
        final_conv_bias_initializer = initializers['final_conv_bias_initializer']
        fcn_reg_h, fcn_reg_angle_cls = None, None
        with tf.variable_scope('reg_conv2d'+scope_postfix):
            fcn_reg_conv2d = inputs
            #activation_fn=tf.nn.leaky_relu
            activation_fn=tf.nn.relu
            for i in range(4):
                fcn_reg_conv2d = slim.conv2d(inputs=fcn_reg_conv2d,
                        num_outputs=self._fpn_channel,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=activation_fn,
                        weights_initializer=subnet_weights_initializer,
                        biases_initializer=subnet_bias_initializer,
                        scope='sub{}'.format(i),)
                tf.summary.histogram('histogram_'+fcn_reg_conv2d.name.replace(':', '_'),
                    fcn_reg_conv2d)

        with tf.variable_scope('output'+scope_postfix):
            fcn_reg_boxes = slim.conv2d(inputs=fcn_reg_conv2d,
                    num_outputs=5 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1,
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=subnet_bias_initializer,
                    scope='boxes',
                    activation_fn=None,)
            tf.summary.histogram('histogram_'+fcn_reg_boxes.name.replace(':', '_'),
                fcn_reg_boxes)
            fcn_reg_boxes = tf.reshape(fcn_reg_boxes, [-1, 5], name='boxes/reshape')
        if add_angle:
            with tf.variable_scope('ang_conv2d'+scope_postfix):
                fcn_ang_conv2d = inputs
                #activation_fn=tf.nn.leaky_relu
                activation_fn=tf.nn.relu
                for i in range(4):
                    fcn_ang_conv2d = slim.conv2d(inputs=fcn_ang_conv2d,
                        num_outputs=self._fpn_channel,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=activation_fn,
                        weights_initializer=subnet_weights_initializer,
                        biases_initializer=subnet_bias_initializer,
                        scope='sub{}'.format(i),)
                    tf.summary.histogram('histogram_'+fcn_ang_conv2d.name.replace(':', '_'),
                        fcn_ang_conv2d)
                    #biases_initializer=subnet_bias_initializer,
            with tf.variable_scope('output_angle_cls'+scope_postfix):
                fcn_reg_angle_cls = slim.conv2d(inputs=fcn_ang_conv2d,
                    num_outputs=2 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1, 
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=final_conv_bias_initializer,
                    activation_fn=None,)
                tf.summary.histogram('histogram_'+fcn_reg_angle_cls.name.replace(':', '_'),
                        fcn_reg_angle_cls)
                fcn_reg_angle_cls = tf.reshape(fcn_reg_angle_cls, [-1, 2], name='angle_cls/reshape')

        if add_h:
            with tf.variable_scope('h_conv2d'+scope_postfix):
                fcn_h_conv2d = inputs
                #activation_fn=tf.nn.leaky_relu
                activation_fn=tf.nn.relu
                for i in range(4):
                    fcn_h_conv2d = slim.conv2d(inputs=fcn_h_conv2d,
                        num_outputs=self._fpn_channel,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=activation_fn,
                        weights_initializer=subnet_weights_initializer,
                        biases_initializer=subnet_bias_initializer,
                        scope='sub{}'.format(i),)
                    tf.summary.histogram('histogram_'+fcn_h_conv2d.name.replace(':', '_'),
                        fcn_h_conv2d)
            with tf.variable_scope('output_h'+scope_postfix):
                fcn_reg_h = slim.conv2d(inputs=fcn_h_conv2d,
                    num_outputs=2 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1, 
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=subnet_bias_initializer,
                    activation_fn=None,)
                tf.summary.histogram('histogram_'+fcn_reg_h.name.replace(':', '_'),
                        fcn_reg_h)
                fcn_reg_h = tf.reshape(fcn_reg_h, [-1, 2], name='h3d/reshape')
            #num of angle classes is 2, one for head and another for tail
        return fcn_reg_boxes, fcn_reg_h, fcn_reg_angle_cls



    def _fcn_reg_net_sharedv1(self, inputs, scope_postfix, initializers, num_anchors_per_location, add_h, add_angle):
        subnet_weights_initializer = initializers['subnet_weights_initializer']
        subnet_bias_initializer = initializers['subnet_bias_initializer']
        final_conv_bias_initializer = initializers['final_conv_bias_initializer']
        fcn_reg_h, fcn_reg_angle_cls = None, None
        with tf.variable_scope('reg_conv2d'+scope_postfix):
            fcn_reg_conv2d = inputs
            #activation_fn=tf.nn.leaky_relu
            activation_fn=tf.nn.relu
            for i in range(4):
                fcn_reg_conv2d = slim.conv2d(inputs=fcn_reg_conv2d,
                        num_outputs=self._fpn_channel,
                        kernel_size=[3, 3],
                        stride=1,
                        activation_fn=activation_fn,
                        weights_initializer=subnet_weights_initializer,
                        biases_initializer=subnet_bias_initializer,
                        scope='sub{}'.format(i),)
                tf.summary.histogram('histogram_'+fcn_reg_conv2d.name.replace(':', '_'),
                    fcn_reg_conv2d)
        with tf.variable_scope('output'+scope_postfix):
            fcn_reg_boxes = slim.conv2d(inputs=fcn_reg_conv2d,
                    num_outputs=5 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1,
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=subnet_bias_initializer,
                    scope='boxes',
                    activation_fn=None,)
            tf.summary.histogram('histogram_'+fcn_reg_boxes.name.replace(':', '_'),
                fcn_reg_boxes)
            fcn_reg_boxes = tf.reshape(fcn_reg_boxes, [-1, 5], name='boxes/reshape')
            if add_h:
                fcn_reg_h = slim.conv2d(inputs=fcn_reg_conv2d,
                    num_outputs=2 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1, 
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=subnet_bias_initializer,
                    scope='h3d',
                    activation_fn=None,)
                tf.summary.histogram('histogram_'+fcn_reg_h.name.replace(':', '_'),
                        fcn_reg_h)
                fcn_reg_h = tf.reshape(fcn_reg_h, [-1, 2], name='h3d/reshape')
            #num of angle classes is 2, one for head and another for tail
            if add_angle:
                    #biases_initializer=subnet_bias_initializer,
                fcn_reg_angle_cls = slim.conv2d(inputs=fcn_reg_conv2d,
                    num_outputs=2 * num_anchors_per_location,
                    kernel_size=[3, 3],
                    stride=1, 
                    weights_initializer=subnet_weights_initializer,
                    biases_initializer=final_conv_bias_initializer,
                    scope='angle_cls',
                    activation_fn=None,)
                tf.summary.histogram('histogram_'+fcn_reg_angle_cls.name.replace(':', '_'),
                        fcn_reg_angle_cls)
                fcn_reg_angle_cls = tf.reshape(fcn_reg_angle_cls, [-1, 2], name='angle_cls/reshape')

        return fcn_reg_boxes, fcn_reg_h, fcn_reg_angle_cls

    def _refine_stage(self, feature_pyramids, anchors_list, reg_boxes_list, cls_probs_list, \
            refine_stage_idx, initializers):
            #argmax_anchor_indices=None):
        final_indices_list_pred = []
        refine_feature_pyramids = {}
        refine_scores_list, refine_probs_list = [], []
        refine_boxes_delta_list = []
        refine_boxes_h_list = []
        refine_boxes_angle_cls_list = []
        pred_boxes_list = []
        #prepare all level
        if refine_stage_idx == 0:
            all_level_size = [0] + self._num_anchors_per_level
            all_level_idx = np.cumsum(all_level_size)
            #empty anchor convert to levelwise
            empty_anchor_filter = self.placeholders[self.PL_ANCHOR_MASK]
            all_level_empty_anchor_filter = [\
                    empty_anchor_filter[all_level_idx[i]:all_level_idx[i+1]]\
                    for i in range(len(self._num_anchors_per_level))]
            num_loc_per_level = [int(n/self._num_anchors_per_location) for n in self._num_anchors_per_level]
            #for shift indices_for_pred from 0 to each level's start
            #e.g. indices at each level: level0: (0~100), level1: (0~30), level2: (0~10) 
            #=> final_indices at all level: concat( (0~100), (100~130), (130~140) )
            offset_location_size = np.cumsum(num_loc_per_level)
        for l, level in enumerate(self._feature_pyramid_levels): 
            with tf.variable_scope(level):
                reg_offsets = tf.reshape(reg_boxes_list[l], (-1, 5))
                anchors = anchors_list[l]
                anchor_stride = self._anchor_params['anchor_strides'][l]
                if refine_stage_idx == 0:
                    #input anchors shape: (feat_h*feat_w*num_anchor_per_loc, )
                    #choose max prob for eacn anchor location.
                    #shape: (feat_h*feat_w*num_anchor_per_loc, )
                    empty_anchor_mask = all_level_empty_anchor_filter[l] 
                    non_empty_anchor_indices = tf.where(empty_anchor_mask)
                    empty_anchor_mask = tf.reshape(empty_anchor_mask, \
                            (-1, self._num_anchors_per_location))
                    #shape: (feat_h*feat_w*num_anchor_per_loc*num_classes, )
                    cls_probs = cls_probs_list[l]
                    cls_probs = tf.reshape(cls_probs,
                        (-1, self._num_anchors_per_location, self.dataset.num_classes))
                    cls_max_probs = tf.reduce_max(cls_probs, axis=-1)
                    #shape: (feat_h*feat_w, self._num_anchor_per_loc)
                    cls_max_probs = tf.where(empty_anchor_mask, cls_max_probs, tf.zeros_like(cls_max_probs))
                    argmax_anchor_idx = tf.cast(tf.argmax(cls_max_probs, axis=-1), tf.int32)
                    argmax_anchor_row_idx = tf.range(tf.shape(cls_probs)[0])
                    indices = tf.stack([argmax_anchor_row_idx, argmax_anchor_idx], axis=1)
                    indices = tf.cast(indices, tf.int64)
                    dense_shape = tf.cast(tf.shape(empty_anchor_mask), tf.int64)
                    indices_sp = tf.SparseTensor(indices, values=tf.ones_like(argmax_anchor_idx), \
                            dense_shape=dense_shape)
                    argmax_mask = tf.sparse.to_dense(indices_sp, default_value=0)
                    argmax_mask = tf.cast(argmax_mask, tf.bool)
                    argmax_mask = tf.reshape(argmax_mask, (-1,))
                    argmax_anchors = tf.boolean_mask(anchors, argmax_mask)
                    argmax_offsets = tf.boolean_mask(reg_offsets, argmax_mask)
                    pred_boxes = anchor_bev_encoder.offset_to_anchor(\
                            argmax_anchors, argmax_offsets)
                    #indices use for GT when compting loss
                    empty_anchor_mask = tf.reshape(empty_anchor_mask, (-1,))
                    #shape (feat_h*feat_w* num_anchors_per_loc)
                    final_mask = tf.logical_and(empty_anchor_mask, argmax_mask)
                    #mask for network output, which shape is (feat_h*feat_w, )
                    final_mask_int = tf.cast(tf.reshape(final_mask, (-1, self._num_anchors_per_location))\
                            , tf.int32)
                    #reshape and reduce sum to get shape (feat_h*feat_w, ) from(feat_h*feat_w*num_anchors_per_loc)
                    final_mask_for_pred = tf.reduce_sum(final_mask_int, axis=-1)
                    final_mask_for_pred = tf.cast(final_mask_for_pred, tf.bool)
                    final_indices_for_pred = tf.where(final_mask_for_pred) #return of where shape is (-1, 1) 
                    num_loc = int(self._num_anchors_per_level[l] / self._num_anchors_per_location)
                    final_indices_for_pred = tf.reshape(final_indices_for_pred, (-1, ))
                    #DEBUG to check final indices for pred == range(feat_h*feat_w) 
                     #when empty_anchor_filter=ALL TRUE
                    #all_anchor_indices = tf.cast(tf.range(num_loc), tf.int32)
                    #diff = tf.subtract(tf.cast(final_indices_for_pred, tf.int32),\
                    #        all_anchor_indices)
                    #final_indices_for_pred = tf.Print(final_indices_for_pred,\
                    #        [tf.reduce_sum(diff), final_indices_for_pred[-3:]], f'refine {num_loc} boxes, indices diff ')
                    if l > 0:
                        final_indices_for_pred += \
                                offset_location_size[l-1]
                                #WRONG before!!!!
                                #int(self._num_anchors_per_level[l-1] / self._num_anchors_per_location)
                        #DEBUG to check if the head and tail is expected indices.
                        #final_indices_for_pred = tf.Print(final_indices_for_pred,\
                        #        [final_indices_for_pred[0:3], final_indices_for_pred[-3:]], f'refine {num_loc} boxes, indices ')
                    final_indices_list_pred.append(final_indices_for_pred)
                else:
                    pred_boxes = anchor_bev_encoder.offset_to_anchor(\
                            anchors, reg_offsets)
                #shape of pred_boxes: (feat_h*feat_w, )
                center_points = pred_boxes[:, :2] / anchor_stride   
                with tf.variable_scope('refine_feature'):
                    refine_feature_pyramids[level] = refine_feature_op(
                        points=center_points, feature_map=feature_pyramids[level],
                        output_channel=self._fpn_channel, initializers=initializers)
                pred_boxes_list.append(pred_boxes)

            share_net = True
            if share_net:
                reuse_flag = None if level == self._feature_pyramid_levels[0] else True
            else:
                reuse_flag = None
            with tf.variable_scope('refine_cls_net', reuse=reuse_flag):
                fcn_cls_conv2d = refine_feature_pyramids[level]
                scope_postfix = f'_{level}' if not share_net else ''
                fcn_cls_scores, fcn_cls_probs = \
                    self._fcn_cls_net(\
                        inputs=fcn_cls_conv2d, scope_postfix=scope_postfix,
                        initializers=initializers,
                        num_anchors_per_location=1)
                refine_scores_list.append(fcn_cls_scores)
                refine_probs_list.append(fcn_cls_probs)
            with tf.variable_scope('refine_reg_net', reuse=reuse_flag):
                fcn_reg_conv2d = refine_feature_pyramids[level]
                scope_postfix = f'_{level}' if not share_net else ''
                fcn_reg_boxes, fcn_reg_h, fcn_reg_angle_cls =\
                    self._fcn_reg_net(\
                        inputs=fcn_reg_conv2d, scope_postfix=scope_postfix,
                        initializers=initializers,
                        num_anchors_per_location=1,
                        add_h=self.add_h_flags[refine_stage_idx+1],
                        add_angle=self.add_angle_flags[refine_stage_idx+1])
                refine_boxes_delta_list.append(fcn_reg_boxes)
                refine_boxes_h_list.append(fcn_reg_h)
                refine_boxes_angle_cls_list.append(fcn_reg_angle_cls)
                    
        refine_results = {
                self.NET_CLS_SCORES_LIST: refine_scores_list,
                self.NET_CLS_PROBS_LIST: refine_probs_list,
                self.NET_REG_BOXES_LIST: refine_boxes_delta_list,
                self.NET_REG_H_LIST: refine_boxes_h_list,
                self.NET_REG_ANGLE_CLS_LIST: refine_boxes_angle_cls_list,
                self.PRED_BOXES_LIST: pred_boxes_list,
            }
        if refine_stage_idx == 0:
            refine_results[self.NET_ANCHOR_INDICES] = \
                    tf.concat(final_indices_list_pred, axis=0)

        return refine_results


    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_feature_pyramids = self.bev_feature_pyramids
        img_feature_pyramids = self.img_feature_pyramids

        #summary_utils.add_feature_maps_from_dict(bev_feature_pyramids, self._feature_pyramid_levels[0])

        # TODO: move this section into an separate AnchorPredictor class
        with tf.variable_scope('anchor_predictor', 'ap', bev_feature_pyramids):
            #tensor_in = bev_proposal_input
            tensor_in_dict = bev_feature_pyramids

            weight_decay = 1e-4
            if weight_decay > 0:
                weights_regularizer = slim.l2_regularizer(weight_decay)
            else:
                weights_regularizer = None

            subnet_weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None)
            subnet_bias_initializer = tf.constant_initializer(value=0.0)
            final_conv_bias_initializer = tf.constant_initializer(value=-np.log((1.0-0.01)/0.01))
            initializers = {
                    'subnet_weights_initializer':subnet_weights_initializer,
                    'subnet_bias_initializer':subnet_bias_initializer,
                    'final_conv_bias_initializer':final_conv_bias_initializer,
            }

            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=weights_regularizer):
                # Use conv2d instead of fully_connected layers.
                fcn_scores_list = []
                fcn_probs_list = []
                fcn_boxes_delta_list = []
                fcn_boxes_h_list = []
                fcn_boxes_angle_cls_list = []
                share_net = True
                for l, level in enumerate(self._feature_pyramid_levels):
                    #FCN for each level
                    if share_net:
                        reuse_flag = None if level == self._feature_pyramid_levels[0] else True
                    else:
                        reuse_flag = None
                    #cls net
                    print('building level : ', level)
                    print('Reuse flag: ', reuse_flag)
                    with tf.variable_scope('fcn_cls_net', reuse=reuse_flag):
                        fcn_cls_conv2d = tensor_in_dict[level]
                        scope_postfix = f'_{level}' if not share_net else ''
                        fcn_cls_scores, fcn_cls_probs = self._fcn_cls_net(\
                                inputs=fcn_cls_conv2d, scope_postfix=scope_postfix,
                                initializers=initializers,
                                num_anchors_per_location=self._num_anchors_per_location)
                        fcn_scores_list.append(fcn_cls_scores)
                        fcn_probs_list.append(fcn_cls_probs)
                    #reg net
                    with tf.variable_scope('fcn_reg_net', reuse=reuse_flag):
                        fcn_reg_conv2d = tensor_in_dict[level]
                        scope_postfix = f'_{level}' if not share_net else ''
                        fcn_reg_boxes, fcn_reg_h, fcn_reg_angle_cls = \
                            self._fcn_reg_net(\
                                inputs=fcn_reg_conv2d, scope_postfix=scope_postfix,
                                initializers=initializers,
                                num_anchors_per_location=self._num_anchors_per_location,
                                add_h=self.add_h_flags[0], add_angle=self.add_angle_flags[0])
                        fcn_boxes_delta_list.append(fcn_reg_boxes)
                        fcn_boxes_h_list.append(fcn_reg_h)
                        fcn_boxes_angle_cls_list.append(fcn_reg_angle_cls)
        fcn_outputs = {
                self.NET_CLS_SCORES_LIST: fcn_scores_list,
                self.NET_CLS_PROBS_LIST: fcn_probs_list,
                self.NET_REG_BOXES_LIST: fcn_boxes_delta_list,
                self.NET_REG_H_LIST: fcn_boxes_h_list,
                self.NET_REG_ANGLE_CLS_LIST: fcn_boxes_angle_cls_list,
                self.NET_ANCHOR_INDICES: self.placeholders[self.PL_ANCHOR_INDICES]}


        if self.refine_stage_num > 0:
            #num_anchor_per_location = 1 instead of self._num_anchor_per_location
            input_reg_boxes_list = fcn_boxes_delta_list
            input_cls_probs_list = fcn_probs_list
            anchors = self.placeholders[self.PL_ANCHORS]
            all_level_size = [0] + self._num_anchors_per_level
            all_level_idx = np.cumsum(all_level_size)
            all_level_anchors = [\
                    anchors[all_level_idx[i]:all_level_idx[i+1]]\
                    for i in range(len(self._num_anchors_per_level))]
            input_anchors_list = all_level_anchors
            refine_stage_outputs = []
            refine_stage_targets = []
            #logical_and(non empty anchor, argmax_anchor at num_anchor_per_loc)
            final_indices_pred = None
            gt_anchors = self._prepare_gt_anchors()
            for i in range(self.refine_stage_num):
                with tf.variable_scope('refine_stage_{}'.format(i), [bev_feature_pyramids]):
                    print('building refine stage{}'.format(i))
                    refine_outputs = \
                        self._refine_stage(\
                            feature_pyramids=bev_feature_pyramids,
                            anchors_list=input_anchors_list,
                            reg_boxes_list=input_reg_boxes_list,
                            cls_probs_list=input_cls_probs_list,
                            refine_stage_idx=i, initializers=initializers)
                    refine_stage_outputs.append(refine_outputs)
                    if i == 0:
                        final_indices_pred = refine_outputs[self.NET_ANCHOR_INDICES]
                    input_anchors_list = refine_outputs[self.PRED_BOXES_LIST]
                    input_reg_boxes_list = refine_outputs[self.NET_REG_BOXES_LIST]
                    input_cls_probs_list = refine_outputs[self.NET_CLS_PROBS_LIST]
                    if self._train_val_test in ['train', 'val']:
                        with tf.variable_scope('build_targets'):
                            print('building refine stage{} targets'.format(i))
                            refine_anchors = tf.concat(\
                                    refine_outputs[self.PRED_BOXES_LIST], axis=0)
                            targets = self._build_refine_targets(\
                                    gt_anchors, refine_anchors,\
                                    refine_stage_idx=i,\
                                    add_h=self.add_h_flags[i+1],\
                                    add_angle=self.add_angle_flags[i+1])
                        refine_stage_targets.append(targets)

        #Add histogram of all trainable vars.
        with tf.variable_scope('histogram'):
            for var in slim.get_model_variables():
                if var.name.find('weights') == -1 and \
                        var.name.find('biases') == -1:
                    continue
                if var.name.find('resnet') != -1 and\
                    var.name.find('fuse_P') == -1:
                        continue
                tf.summary.histogram(var.name.replace(':', '_'),
                                 var)
        

        anchors = self.placeholders[self.PL_ANCHORS]
        #Non empty indices
        non_empty_anchor_indices = self.placeholders[self.PL_ANCHOR_INDICES]

        # Get mini batch
        all_ious_gt = self.placeholders[self.PL_ANCHOR_IOUS]
        all_offsets_gt = self.placeholders[self.PL_ANCHOR_OFFSETS]
        all_offsets_h_gt = self.placeholders[self.PL_ANCHOR_OFFSETS_H]
        all_offsets_angle_cls_gt = self.placeholders[self.PL_ANCHOR_OFFSETS_ANGLE_CLS]
        all_classes_gt = self.placeholders[self.PL_ANCHOR_CLASSES]
       # Ground Truth Tensors
        with tf.variable_scope('fcn_positive_anchor'):
            #positive anchor for FCN (before any refine stages)
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.retinanet_pos_iou_range[0]
            max_neg_iou = \
                self.dataset.kitti_utils.mini_batch_utils.retinanet_neg_iou_range[1]
            pos_anchor_mask = tf.greater_equal(all_ious_gt, min_pos_iou)
            neg_anchor_mask = tf.less(all_ious_gt, max_neg_iou)

        with tf.variable_scope('fcn_build_targets'):

            #pos and neg., 0 is background at all_classes_gt
            objectness_gt_with_bg = tf.one_hot(
                    tf.cast(all_classes_gt, tf.int32),
                    depth=self.dataset.num_classes+1,)
            #filter background, if background, then this row's all cols in objectness_gt is zero.
            objectness_gt = objectness_gt_with_bg[:, 1:]
            #make sure neg anchor's class vector is all zeros
            objectness_gt = tf.where(neg_anchor_mask, tf.zeros_like(objectness_gt), objectness_gt)
            offsets_angle_cls_gt = tf.one_hot(
                tf.cast(all_offsets_angle_cls_gt, tf.int32),
                depth=2,
                )
            fcn_targets = {
                    self.PRED_OBJECTNESS_GT: objectness_gt, 
                    self.PRED_OFFSETS_GT: all_offsets_gt,
                    self.PRED_OFFSETS_H_GT: all_offsets_h_gt,
                    self.PRED_OFFSETS_ANGLE_CLS_GT: offsets_angle_cls_gt,
                    self.POS_ANCHORS_MASK: pos_anchor_mask,
                    self.NEG_ANCHORS_MASK: neg_anchor_mask,
            }
            #gt_anchors = self._prepare_gt_anchors()
            #fcn_targets = self._build_fcn_targets(gt_anchors, anchors, self.add_h_flags[0], self.add_angle_flags[0])
        # Specify the tensors to evaluate
        predictions = dict()

        # Temporary predictions for debugging
        # predictions['anchor_ious'] = anchor_ious
        # predictions['anchor_offsets'] = all_offsets_gt
        # Final NMS results
        top_indices = None
        top_anchors_info = None 
        top_objectness_sigmoid = None
        do_nms = [False] * (1 + self.refine_stage_num)
        do_nms[-1] = True
        #DEBUG so set all stages to True
        #do_nms = [True] * (1 + self.refine_stage_num)
        with tf.variable_scope('fcn_prediction'):
            pred_objectness = tf.concat(fcn_outputs[self.NET_CLS_SCORES_LIST], axis=0) #shape(-1, num_class)
            pred_offsets = tf.concat(fcn_outputs[self.NET_REG_BOXES_LIST], axis=0)
            anchor_indices_for_pred = non_empty_anchor_indices 
            anchor_indices_for_pred = tf.reshape(anchor_indices_for_pred, (-1, 1))
            pred_anchors = tf.gather_nd(anchors, anchor_indices_for_pred) 
            pred_objectness = tf.gather_nd(pred_objectness, anchor_indices_for_pred)
            pred_offsets = tf.gather_nd(pred_offsets, anchor_indices_for_pred)

            pred_offsets_h = None
            if self.add_h_flags[0]:
                pred_offsets_h = tf.concat(fcn_outputs[self.NET_REG_H_LIST], axis=0)
                pred_offsets_h = tf.gather_nd(pred_offsets_h, anchor_indices_for_pred)
            pred_offsets_angle_cls = None
            if self.add_angle_flags[0]:
                pred_offsets_angle_cls = tf.concat(fcn_outputs[self.NET_REG_ANGLE_CLS_LIST], axis=0)
                pred_offsets_angle_cls = tf.gather_nd(pred_offsets_angle_cls, anchor_indices_for_pred)

            fcn_pred_results = {self.PRED_OBJECTNESS: pred_objectness,
                    self.PRED_OFFSETS: pred_offsets,
                    self.PRED_OFFSETS_H: pred_offsets_h,
                    self.PRED_OFFSETS_ANGLE_CLS: pred_offsets_angle_cls,
                    self.PRED_ANCHORS: pred_anchors,
                    }
            if self._train_val_test in ['train', 'val']:
                #fcn targets already filtered by non empty anchor indices at preprocessing stage
                fcn_pred_results.update(fcn_targets)
                vis_pos_anchor = False
                if vis_pos_anchor:
                    with tf.variable_scope('vis_fcn_pos_anchor'):
                        self.visualize_positive_anchor(anchors=pred_anchors, \
                                pos_mask=fcn_pred_results[self.POS_ANCHORS_MASK], \
                                objectness=fcn_pred_results[self.PRED_OBJECTNESS],\
                                offsets=fcn_pred_results[self.PRED_OFFSETS],\
                                offsets_gt=fcn_pred_results[self.PRED_OFFSETS_GT],\
                                offsets_h=fcn_pred_results[self.PRED_OFFSETS_H])
            do_nms[0] = True
            if do_nms[0] and self.do_nms_at_gpu:
                nms_result = self.get_nms_result(fcn_pred_results, visualize=True)
                if self.refine_stage_num == 0:
                    top_anchors_info = nms_result[self.PRED_TOP_ANCHORS]
                    top_objectness_sigmoid = nms_result[self.PRED_TOP_OBJECTNESS_SIGMOID]
                    top_indices = nms_result[self.PRED_TOP_INDICES]
            elif self.refine_stage_num == 0:
                #save all pred result as top result, for NMS at CPU later
                regressed_anchors = anchor_bev_encoder.offset_to_anchor(
                    fcn_pred_results[self.PRED_ANCHORS], fcn_pred_results[self.PRED_OFFSETS])
                top_anchors_info = tf.concat([
                    regressed_anchors,
                    fcn_pred_results[self.PRED_OFFSETS_H], 
                    fcn_pred_results[self.PRED_OFFSETS_ANGLE_CLS]], axis=1)
                top_objectness_sigmoid = tf.sigmoid(fcn_pred_results[self.PRED_OBJECTNESS], name='sigmoid') 
                top_objectness_sigmoid = tf.reshape(top_objectness_sigmoid, (-1,))
                top_indices = tf.range(tf.shape(top_anchors_info)[0])

        refine_stage_pred_results = []
        for i in range(self.refine_stage_num):
            with tf.variable_scope(f'refine_{i}_prediction'):
                refine_outputs = refine_stage_outputs[i]
                pred_anchors = tf.concat(refine_outputs[self.PRED_BOXES_LIST], axis=0)
                pred_objectness = tf.concat(refine_outputs[self.NET_CLS_SCORES_LIST], axis=0) #shape(-1, num_class)
                pred_offsets = tf.concat(refine_outputs[self.NET_REG_BOXES_LIST], axis=0)
                anchor_indices_for_pred = final_indices_pred 
                anchor_indices_for_pred = tf.reshape(anchor_indices_for_pred, (-1, 1))
                pred_anchors = tf.gather_nd(pred_anchors, anchor_indices_for_pred)
                pred_objectness = tf.gather_nd(pred_objectness, anchor_indices_for_pred)
                pred_offsets = tf.gather_nd(pred_offsets, anchor_indices_for_pred)
                pred_offsets_h = None
                if self.add_h_flags[i+1]:
                    pred_offsets_h = tf.concat(refine_outputs[self.NET_REG_H_LIST], axis=0)
                    pred_offsets_h = tf.gather_nd(pred_offsets_h, anchor_indices_for_pred)
                pred_offsets_angle_cls = None
                if self.add_angle_flags[i+1]:
                    pred_offsets_angle_cls = tf.concat(refine_outputs[self.NET_REG_ANGLE_CLS_LIST], axis=0)
                    pred_offsets_angle_cls = tf.gather_nd(pred_offsets_angle_cls, anchor_indices_for_pred)

                refine_pred_results = {self.PRED_OBJECTNESS: pred_objectness,
                        self.PRED_OFFSETS: pred_offsets,
                        self.PRED_OFFSETS_H: pred_offsets_h,
                        self.PRED_OFFSETS_ANGLE_CLS: pred_offsets_angle_cls,
                        self.PRED_ANCHORS: pred_anchors,
                        }
                #refine gather gt.
                if self._train_val_test in ['train', 'val']:
                    targets = refine_stage_targets[i]
                    #has gt label
                    refine_objectness_gt = targets[self.PRED_OBJECTNESS_GT]
                    refine_offsets_gt = targets[self.PRED_OFFSETS_GT]
                    refine_offsets_angle_cls_gt = targets[self.PRED_OFFSETS_ANGLE_CLS_GT]
                    refine_offsets_h_gt = targets[self.PRED_OFFSETS_H_GT]
                    refine_pos_mask = targets[self.POS_ANCHORS_MASK]
                    refine_neg_mask = targets[self.NEG_ANCHORS_MASK]
                    refine_objectness_gt = tf.gather_nd(refine_objectness_gt, anchor_indices_for_pred)
                    refine_offsets_gt = tf.gather_nd(refine_offsets_gt, anchor_indices_for_pred)
                    if self.add_h_flags[i+1]:
                        refine_offsets_h_gt = tf.gather_nd(refine_offsets_h_gt, anchor_indices_for_pred)
                    if self.add_angle_flags[i+1]:
                        refine_offsets_angle_cls_gt = tf.gather_nd(refine_offsets_angle_cls_gt, anchor_indices_for_pred)
                    refine_pos_mask = tf.gather_nd(refine_pos_mask, anchor_indices_for_pred)
                    #refine_pos_mask = tf.Print(refine_pos_mask, \
                    #    [tf.reduce_sum(tf.cast(refine_pos_mask, tf.int32)),\
                    #    tf.shape(refine_pos_mask)],\
                    #    'after filter, refine pos num, shape of pos mask')
                    refine_neg_mask = tf.gather_nd(refine_neg_mask, anchor_indices_for_pred)
                    refine_pred_results.update({
                        self.PRED_OBJECTNESS_GT: refine_objectness_gt,
                        self.PRED_OFFSETS_GT: refine_offsets_gt,
                        self.PRED_OFFSETS_H_GT: refine_offsets_h_gt,
                        self.PRED_OFFSETS_ANGLE_CLS_GT: refine_offsets_angle_cls_gt,
                        self.POS_ANCHORS_MASK: refine_pos_mask,
                        self.NEG_ANCHORS_MASK: refine_neg_mask,
                    })
                    #visualize positive anchor
                    vis_pos_anchor = True
                    if vis_pos_anchor:
                        with tf.variable_scope(f'vis_refine{i}_pos_anchor'):
                            self.visualize_positive_anchor(\
                                    anchors=pred_anchors, \
                                    pos_mask=refine_pred_results[self.POS_ANCHORS_MASK],\
                                    objectness=refine_pred_results[self.PRED_OBJECTNESS],\
                                    offsets=refine_pred_results[self.PRED_OFFSETS],\
                                    offsets_gt=refine_pred_results[self.PRED_OFFSETS_GT],\
                                    offsets_h=refine_pred_results[self.PRED_OFFSETS_H])
                refine_stage_pred_results.append(refine_pred_results)
                if do_nms[i+1]:
                    nms_result = self.get_nms_result(refine_pred_results, visualize=True)
                    if i == self.refine_stage_num - 1:
                        top_anchors_info = nms_result[self.PRED_TOP_ANCHORS]
                        top_objectness_sigmoid = nms_result[self.PRED_TOP_OBJECTNESS_SIGMOID]
                        top_indices = nms_result[self.PRED_TOP_INDICES]

        if self._train_val_test in ['train', 'val']:
            # All anchors
            #predictions[self.PRED_ANCHORS] = anchors
            stage_keys = ['fcn'] + [f'refine{i}' for i in range(self.refine_stage_num)]
            all_pred_results = [fcn_pred_results] + refine_stage_pred_results
            #for i in range(len(stage_keys)):
            for i, sk in enumerate(self.STAGE_KEYS):
                pred_results = all_pred_results[i]
                stage_predictions = dict()
                stage_predictions[self.POS_ANCHORS_MASK] = pred_results[self.POS_ANCHORS_MASK]
                stage_predictions[self.NEG_ANCHORS_MASK] = pred_results[self.NEG_ANCHORS_MASK] 
                # predictions, masked by non empty anchors' indices.
                stage_predictions[self.PRED_OBJECTNESS] = pred_results[self.PRED_OBJECTNESS] #objectness
                stage_predictions[self.PRED_OFFSETS] = pred_results[self.PRED_OFFSETS] #offsets
                stage_predictions[self.PRED_OFFSETS_GT] = pred_results[self.PRED_OFFSETS_GT] 
                stage_predictions[self.PRED_OBJECTNESS_GT] = pred_results[self.PRED_OBJECTNESS_GT] 
 
                if self.add_h_flags[i]:
                    stage_predictions[self.PRED_OFFSETS_H] = pred_results[self.PRED_OFFSETS_H] #offsets_h
                    stage_predictions[self.PRED_OFFSETS_H_GT] = pred_results[self.PRED_OFFSETS_H_GT] 
                if self.add_angle_flags[i]:
                    stage_predictions[self.PRED_OFFSETS_ANGLE_CLS] = pred_results[self.PRED_OFFSETS_ANGLE_CLS]             
                    stage_predictions[self.PRED_OFFSETS_ANGLE_CLS_GT] = pred_results[self.PRED_OFFSETS_ANGLE_CLS_GT] 
                predictions[sk] = stage_predictions

           # Proposals after nms
            predictions[self.PRED_TOP_INDICES] = top_indices
            predictions[self.PRED_TOP_ANCHORS] = top_anchors_info
            predictions[
                self.PRED_TOP_OBJECTNESS_SIGMOID] = top_objectness_sigmoid

        else:
            # self._train_val_test == 'test'
            raise NotImplementedError('Not implementation for inference yet.')
            #predictions[self.PRED_TOP_ANCHORS] = top_anchors
            #predictions[
            #    self.PRED_TOP_OBJECTNESS_SIGMOID] = top_objectness_sigmoid

        return predictions

    
    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None and self._train_val_test in ['train']:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    samples = self.dataset.next_batch(batch_size=1)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=False)

                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When #training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if anchors_info or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)
        #print('SAMPLE NAME: ', sample_name)

        # Get ground truth data
        label_classes = sample.get(constants.KEY_LABEL_CLASSES)
        # We only need orientation from box_3d
        label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)
        bev_input = sample.get(constants.KEY_BEV_INPUT) #all height maps and density
        bev_input = bev_input[:, :, [-1, 0, 1]] #height0, height2, density for dbg4 and all after.
        #bev_input = bev_input[:, :, [-1, 0, 2]] #height0, height2, density

        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        ground_plane = sample.get(constants.KEY_GROUND_PLANE)
        stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)

        #WZN: sparse pooling input
        sparse_pooling_inputs = sample.get(constants.KEY_SPARSE_POOLING_INPUT)

        # Fill the placeholders for anchor information
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,
                                    ground_plane=ground_plane,
                                    image_shape=image_shape,
                                    stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name,
                                    sample_augs=sample_augs)

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        self._placeholder_inputs[self.PL_IMG_INPUT] = image_input

        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['anchors_info'] = anchors_info

        #WZN: sparse pooling info
        if self._use_sparse_pooling:
            sparse_pooling_input1 = sparse_pooling_inputs[0]
            self._placeholder_inputs[self.PL_M_VAL] = sparse_pooling_input1['M_val']
            self._placeholder_inputs[self.PL_M_IJ] = sparse_pooling_input1['Mij_pool']
            self._placeholder_inputs[self.PL_M_SIZE] = sparse_pooling_input1['M_size']
            self._placeholder_inputs[self.PL_IMG_POOL_IJ] = sparse_pooling_input1['img_index_flip_pool']
            self._placeholder_inputs[self.PL_BEV_POOL_IJ] = sparse_pooling_input1['bev_index_flip_pool']
        else:
            self._placeholder_inputs[self.PL_M_VAL] = np.zeros((0))
            self._placeholder_inputs[self.PL_M_IJ] = np.zeros((0,2))
            self._placeholder_inputs[self.PL_M_SIZE] = np.zeros((2))
            self._placeholder_inputs[self.PL_IMG_POOL_IJ] = np.zeros((0,3))
            self._placeholder_inputs[self.PL_BEV_POOL_IJ] = np.zeros((0,3))

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # Lists for merging anchors info
        all_anchor_boxes_bev = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        all_level_anchor_boxes_bev = self._anchor_generator.generate(\
                image_shapes=self._anchor_params['image_shapes'],
                anchor_base_sizes=self._anchor_params['anchor_base_sizes'],
                anchor_strides=self._anchor_params['anchor_strides'],
                anchor_ratios=self._anchor_params['anchor_ratios'],
                anchor_scales=self._anchor_params['anchor_scales'],
                anchor_init_ry_type=self._anchor_params['anchor_init_ry_type'])
        #print('anchor params for generator:\n', self._anchor_params)
        #concate all levels anchors
        all_anchor_boxes_bev = np.concatenate(all_level_anchor_boxes_bev)
        #print('anchors shape from anchor_generator: ', all_anchor_boxes_bev.shape)

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchors_ious, anchor_offsets, \
                        anchor_classes = anchors_info

                #anchor_boxes_bev_to_use = all_anchor_boxes_bev[anchor_indices]
                empty_anchor_filter = np.zeros(all_anchor_boxes_bev.shape[0], \
                        dtype=np.bool)
                empty_anchor_filter[anchor_indices] = True
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            anchors_bev = all_anchor_boxes_bev
            #anchors_bev = all_anchor_boxes_bev.copy()
            #if self._anchor_params['anchor_init_ry_type'] == -90:
            #    anchors_bev[:, [2, 3]] = anchors_bev[:, [3,2]] 
            anchors_3d = box_bev_encoder.box_bev_to_anchor_3d(anchors_bev, \
                    bev_shape=self._bev_pixel_size, \
                    bev_extents=self._bev_extents)
            # Generate sliced 2D voxel grid for filtering
            vx_grid_2d = self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                sample_name, source=self.dataset.bev_source,
                image_shape=image_shape)
            empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_3d, vx_grid_2d, density_threshold=1)
            #print(f'Non empty anchor: {np.sum(empty_anchor_filter)} / {len(anchor_boxes_bev)}, sample_name: {sample_name}')
            anchor_indices = np.where(empty_anchor_filter)[0]
            
        #anchor_boxes_bev_to_use = all_anchor_boxes_bev[empty_anchor_filter]
        anchor_boxes_bev_to_use = all_anchor_boxes_bev
        # Convert lists to ndarrays
        anchor_boxes_bev_to_use = np.asarray(anchor_boxes_bev_to_use)
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        if len(anchor_offsets) > 0:
            anchor_offsets_h = anchor_offsets[:, 5:7]
            anchor_offsets_angle_cls = anchor_offsets[:, 7]
            anchor_offsets = anchor_offsets[:, :5]

        anchor_classes = np.asarray(anchor_classes)

        # Convert to anchors
        anchors_to_use = anchor_boxes_bev_to_use
        num_anchors = len(anchors_to_use)
        #print(f'_fill_anchor_pl_inputs: num_anchors: {num_anchors}, anchor_indices.shape: {anchor_indices.shape}')
        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_H] = anchor_offsets_h
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_ANGLE_CLS] = anchor_offsets_angle_cls
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes
            self._placeholder_inputs[self.PL_ANCHOR_INDICES] = anchor_indices
            self._placeholder_inputs[self.PL_ANCHOR_MASK] = empty_anchor_filter

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 5])
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_H] = \
                np.zeros([num_anchors, 2])
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS_ANGLE_CLS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
            #Non empty anchor indices
            self._placeholder_inputs[self.PL_ANCHOR_INDICES] = \
                anchor_indices
            self._placeholder_inputs[self.PL_ANCHOR_MASK] = \
                empty_anchor_filter
            
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))
        #print(f'empty anchor filter , non empty num: {np.sum(empty_anchor_filter)}, total:{empty_anchor_filter.shape[0]}')


    def loss(self, prediction_dict):
        total_loss = 0.
        losses_dict = {}
        #refine loss at each stage
        for i, sk in enumerate(self.STAGE_KEYS):
            stage_predictions = prediction_dict[sk]
            with tf.variable_scope(f'{sk}_loss'):
                slosses_dict, sloss = self.each_loss(stage_predictions, \
                        add_h=self.add_h_flags[i], add_angle=self.add_angle_flags[i])
                #losses_dict[f'refine{i}'] = rlosses_dict
            losses_dict[sk] = slosses_dict
            total_loss += sloss

        return losses_dict, total_loss

    def each_loss(self, prediction_dict, add_h, add_angle):
        # these should include mini-batch values only
        #no groundtruth label for this sample
        objectness_gt = prediction_dict[self.PRED_OBJECTNESS_GT]
        offsets_gt = prediction_dict[self.PRED_OFFSETS_GT]
        if add_h:
            offsets_h_gt = prediction_dict[self.PRED_OFFSETS_H_GT]
        if add_angle:
            offsets_angle_cls_gt = prediction_dict[self.PRED_OFFSETS_ANGLE_CLS_GT]

        pos_anchor_mask = prediction_dict[self.POS_ANCHORS_MASK]
        num_positives = tf.reduce_sum(tf.cast(pos_anchor_mask, tf.int32))
        num_positives = tf.maximum(tf.cast(num_positives, tf.float32), 1.0)
        neg_anchor_mask = prediction_dict[self.NEG_ANCHORS_MASK]
        # Predictions
        with tf.variable_scope('prediction'):
            objectness = prediction_dict[self.PRED_OBJECTNESS]
            offsets = prediction_dict[self.PRED_OFFSETS]
            if add_h:
                offsets_h = prediction_dict[self.PRED_OFFSETS_H]
            if add_angle:
                offsets_angle_cls = prediction_dict[self.PRED_OFFSETS_ANGLE_CLS]

        #pos_anchor_mask = tf.Print(pos_anchor_mask, \
        #        [tf.shape(pos_anchor_mask), num_positives, \
        #        tf.shape(objectness_gt), tf.shape(objectness),\
        #        tf.shape(offsets_gt), tf.shape(offsets), objectness_gt.name],\
        #        '**TF.PRINT** pos_anchor_mask.shape, num_pos, cls_gt, cls, reg_gt, reg,')
        with tf.variable_scope('retinanet_losses'):
            with tf.variable_scope('cls'):
                cls_loss = losses.WeightedFocalLoss()
                cls_loss_weight = self._config.loss_config.cls_loss_weight
                #make sure neg anchor's class vector is all zeros
                #objectness_gt = tf.where(neg_anchor_mask, tf.zeros_like(objectness_gt), objectness_gt)
                #filter out ignore anchor
                pos_or_neg_mask = tf.logical_or(pos_anchor_mask, neg_anchor_mask)
                objectness = tf.boolean_mask(objectness, pos_or_neg_mask)
                objectness_gt = tf.boolean_mask(objectness_gt, pos_or_neg_mask)
                objectness_loss = cls_loss(objectness,
                                           objectness_gt,
                                           weight=cls_loss_weight)
                with tf.variable_scope('norm'):
                    # normalize by the number of anchor, only pos for RetinaNet
                    objectness_loss = objectness_loss / num_positives
                    tf.summary.scalar('val', objectness_loss)

            with tf.variable_scope('reg'):
                reg_loss = losses.WeightedSmoothL1Loss() 
                reg_loss_weight = self._config.loss_config.reg_loss_weight
                offsets_masked = tf.boolean_mask(offsets, pos_anchor_mask)
                offsets_gt_masked = tf.boolean_mask(offsets_gt, pos_anchor_mask)
                anchorwise_localization_loss = reg_loss(offsets_masked,
                                                        offsets_gt_masked,
                                                        weight=reg_loss_weight)
                localization_loss = tf.reduce_sum(anchorwise_localization_loss)
                localization_loss = tf.Print(localization_loss, \
                        [localization_loss, num_positives],
                        f'{localization_loss.name} reg loss, num pos')
                localization_loss = tf.maximum(localization_loss, 0.0)
                if add_angle:
                    with tf.variable_scope('angle_cls'):
                        angle_cls_loss_fn = losses.WeightedSoftmaxLoss()
                        ang_loss_weight = self._config.loss_config.ang_loss_weight
 
                        offsets_angle_cls_masked = tf.boolean_mask(offsets_angle_cls, pos_anchor_mask)
                        offsets_angle_cls_gt_masked = tf.boolean_mask(offsets_angle_cls_gt, pos_anchor_mask)
                        anchorwise_angle_cls_loss = angle_cls_loss_fn(offsets_angle_cls_masked,
                                                                       offsets_angle_cls_gt_masked,
                                                                       weight=ang_loss_weight)
                        angle_cls_loss = tf.reduce_sum(anchorwise_angle_cls_loss)
                        #angle_cls_loss = tf.Print(angle_cls_loss, \
                        #    [angle_cls_loss, num_positives],
                        #    f'{angle_cls_loss.name} ang loss, num pos')
                if add_h:
                    with tf.variable_scope('h_reg'):
                        h_reg_loss_fn = losses.WeightedSmoothL1Loss()
                        h_loss_weight = self._config.loss_config.h_loss_weight
                        offset_h_gt_weight = 1.0
                        offsets_h_masked = tf.boolean_mask(offsets_h, pos_anchor_mask)
                        offsets_h_gt_masked = tf.boolean_mask(offsets_h_gt, pos_anchor_mask)
                        #offsets_h_gt = offsets_h_gt * offset_h_gt_weight
                        anchorwise_h_reg_loss = h_reg_loss_fn(offsets_h_masked,
                                                               offsets_h_gt_masked,
                                                               weight=h_loss_weight)
                        h_reg_loss = tf.reduce_sum(anchorwise_h_reg_loss)
                        #h_reg_loss = tf.Print(h_reg_loss, \
                        #    [h_reg_loss, num_positives],
                        #    f'{h_reg_loss.name} h loss, num pos')
                with tf.variable_scope('norm'):
                    # normalize by the number of positive objects
                    # Assert the condition `num_positives > 0`
                    with tf.control_dependencies(
                            [tf.assert_positive(num_positives)]):

                        localization_loss = localization_loss / num_positives
                        tf.summary.scalar('val', localization_loss)
                        if add_angle:
                            angle_cls_loss = angle_cls_loss / num_positives
                            tf.summary.scalar('angle_cls', angle_cls_loss)
                        if add_h:
                            h_reg_loss = h_reg_loss / num_positives
                            tf.summary.scalar('h_reg', h_reg_loss)

            with tf.variable_scope('total_loss'):
                total_loss = objectness_loss + localization_loss
                if add_h:
                    total_loss += h_reg_loss
                if add_angle:
                    total_loss += angle_cls_loss

        loss_dict = {
            self.LOSS_RETINANET_OBJECTNESS: objectness_loss,
            self.LOSS_RETINANET_REGRESSION: localization_loss,
        }
        if add_h:
            loss_dict[self.LOSS_RETINANET_H] = h_reg_loss
        if add_angle:
            loss_dict[self.LOSS_RETINANET_ANGLE_CLS] = angle_cls_loss
        return loss_dict, total_loss


    def restore_pretrained_backbone_variables(self):
        restore_variables = {}
        resnet_name = self._bev_feature_extractor.config.resnet_name
        load_to_bev, load_to_img = False, False
        if self._bev_feature_extractor.config.load_from_pretrained:
            load_to_bev = True
        if self._img_feature_extractor.config.load_from_pretrained:
            load_to_img = True
        assert(load_to_bev == True or load_to_img == True)
        for var in slim.get_model_variables():
            if var.name.find('resnet') != -1:
                if (load_to_bev and var.name.startswith('bev')) \
                        or (load_to_img and var.name.startswith('img')):
                    var_name_in_ckpt = var.op.name
                    s = var_name_in_ckpt.find('resnet')
                    #convert name from 'img_resnetxxxxxx' to 'resnetxxxxx'
                    var_name_in_ckpt = var_name_in_ckpt[s:] 
                    #print("var_in_graph: {}, var_in_ckpt: {} ".format(var.name, var_name_in_ckpt))
                    restore_variables[var_name_in_ckpt] = var
        return restore_variables, resnet_name

    def _prepare_gt_anchors(self):
        gt_boxes_3d = self.placeholders[self.PL_LABEL_BOXES_3D]
        gt_anchors_norm, _, = tf.py_func(box_3d_projector.project_to_bev_box,
                inp=[gt_boxes_3d, self._bev_extents], Tout=[tf.float32, tf.float32])
        gt_anchors_norm = tf.reshape(gt_anchors_norm, (-1, 5))
        gt_anchors_x = gt_anchors_norm[:, 0] * self._bev_pixel_size[1]
        gt_anchors_y = gt_anchors_norm[:, 1] * self._bev_pixel_size[0]
        gt_anchors_w = gt_anchors_norm[:, 2] * self._bev_pixel_size[1]
        gt_anchors_h = gt_anchors_norm[:, 3] * self._bev_pixel_size[0]
        gt_anchors = tf.stack(\
                [gt_anchors_x, gt_anchors_y, gt_anchors_w, gt_anchors_h, gt_anchors_norm[:, -1]], axis=1)
        return gt_anchors


    def _build_refine_targets(self, gt_anchors, anchors, refine_stage_idx, add_h, add_angle):
        with tf.variable_scope('iou'):
            device_id = 0
            ious = tf.py_func(rotate_iou.calculate_rotate_iou,
                    inp=[gt_anchors, anchors, device_id],
                    Tout=tf.float32)
            max_ious = tf.reduce_max(ious, axis=0)
            max_ious = tf.reshape(max_ious, (-1, ))
            max_iou_indices = tf.argmax(ious, axis=0)
            max_iou_indices = tf.reshape(max_iou_indices, (-1, ))

        with tf.variable_scope(f'positive_anchor'):
            min_pos_iou = \
                self._config.retinanet_config.refine_pos_iou_thresh[refine_stage_idx]
                #self.dataset.kitti_utils.mini_batch_utils.retinanet_pos_iou_range[0]
            max_neg_iou = \
                self._config.retinanet_config.refine_neg_iou_thresh[refine_stage_idx]
                #self.dataset.kitti_utils.mini_batch_utils.retinanet_neg_iou_range[1]
            pos_anchor_mask = tf.greater_equal(max_ious, min_pos_iou)
            neg_anchor_mask = tf.less(max_ious, max_neg_iou)


        with tf.variable_scope('offsets_gt'):
            gt_classes = self.placeholders[self.PL_LABEL_CLASSES]
            gt_boxes_3d = self.placeholders[self.PL_LABEL_BOXES_3D]
            #gt_y3d, gt_h3d
            offsets_h_gt, offsets_angle_cls_gt = None, None
            class_gt = tf.gather(gt_classes, max_iou_indices)
            objectness_gt_with_bg = tf.one_hot(
                    tf.cast(class_gt, tf.int32),
                    depth=self.dataset.num_classes+1,)
            objectness_gt = objectness_gt_with_bg[:, 1:]
            #make sure neg anchor's class vector is all zeros
            objectness_gt = tf.where(neg_anchor_mask, tf.zeros_like(objectness_gt), objectness_gt)
            anchors_gt = tf.gather(gt_anchors, max_iou_indices)
            offsets_gt = anchor_bev_encoder.tf_anchor_to_offset(anchors, anchors_gt)
            if add_h:
                label_h = tf.stack([gt_boxes_3d[:, 1], gt_boxes_3d[:, 5]], axis=1)
                gt_anchor_h = tf.gather(label_h, max_iou_indices)
                #shape: (num_anchor,)
                n_anchor = tf.shape(max_iou_indices)[0]
                anchor_h = anchor_bev_encoder.get_default_anchor_h(n_anchor, fmt='tf')
                offsets_h_gt = anchor_bev_encoder.anchor_to_offset_h(anchor_h, gt_anchor_h)
            if add_angle:
                gt_anchor_angle = tf.gather(gt_anchors[:, -1], max_iou_indices)
                offsets_angle_cls_gt = orientation_encoder.tf_orientation_to_angle_cls(gt_anchor_angle)

        #these info is pred at num anchors = feat_h*feat_w
        #should be filtered later by non empty indices
        targets = {
            self.PRED_OBJECTNESS_GT: objectness_gt,
            self.PRED_OFFSETS_GT: offsets_gt,
            self.PRED_OFFSETS_H_GT: offsets_h_gt,
            self.PRED_OFFSETS_ANGLE_CLS_GT: offsets_angle_cls_gt,
            self.POS_ANCHORS_MASK: pos_anchor_mask,
            self.NEG_ANCHORS_MASK: neg_anchor_mask,
        }
 
        return targets

    def _build_fcn_targets(self, gt_anchors, anchors, add_h, add_angle):
        with tf.variable_scope('iou'):
            iou_type = '2d'
            if iou_type == '2d':
                #anchors_for_2d_iou_h = box_bev_encoder.box_bev_to_iou_h_format(anchors)
                anchors_for_2d_iou_h = tf.py_func(box_bev_encoder.box_bev_to_iou_h_format,
                        inp=[anchors], Tout=tf.float32)
                gt_anchors_for_2d_iou_h = tf.py_func(box_bev_encoder.box_bev_to_iou_h_format,
                        inp=[gt_anchors], Tout=tf.float32)
                anchors_for_2d_iou_h = tf.reshape(anchors_for_2d_iou_h,
                        (-1, 4))
                gt_anchors_for_2d_iou_h = tf.reshape(gt_anchors_for_2d_iou_h,
                        (-1, 4))
                ious = tf.py_func(rotate_iou.calculate_iou,\
                        inp=[gt_anchors_for_2d_iou_h, anchors_for_2d_iou_h],
                        Tout=tf.float32)

                #anchors_h_tf_order = \
                #    anchor_projector.reorder_projected_boxes(anchors_for_2d_iou_h)
                #gt_anchors_h_tf_order = \
                #    anchor_projector.reorder_projected_boxes(gt_anchors_for_2d_iou_h)
                #gt_anchor_box_list = box_list.BoxList(gt_anchors_h_tf_order)
                #anchor_box_list = box_list.BoxList(anchors_h_tf_order)
                #ious = box_list_ops.iou(gt_anchor_box_list, anchor_box_list)
            elif iou_type == '2d_rotate':
                device_id = 0
                ious = tf.py_func(rotate_iou.calculate_rotate_iou,
                        inp=[gt_anchors, anchors, device_id],
                        Tout=tf.float32)
            else:
                raise NotImplementedError(f'Invalid iou_type: {iou_type}, should be [2d, 2d_rotate]')
 
            max_ious = tf.reduce_max(ious, axis=0)
            max_ious = tf.reshape(max_ious, (-1, ))
            max_iou_indices = tf.argmax(ious, axis=0)
            max_iou_indices = tf.reshape(max_iou_indices, (-1, ))

        with tf.variable_scope(f'positive_anchor'):
            min_pos_iou = \
                self.dataset.kitti_utils.mini_batch_utils.retinanet_pos_iou_range[0]
            max_neg_iou = \
                self.dataset.kitti_utils.mini_batch_utils.retinanet_neg_iou_range[1]
            pos_anchor_mask = tf.greater_equal(max_ious, min_pos_iou)
            neg_anchor_mask = tf.less(max_ious, max_neg_iou)


        with tf.variable_scope('offsets_gt'):
            gt_classes = self.placeholders[self.PL_LABEL_CLASSES]
            gt_boxes_3d = self.placeholders[self.PL_LABEL_BOXES_3D]
            #gt_y3d, gt_h3d
            offsets_h_gt, offsets_angle_cls_gt = None, None
            class_gt = tf.gather(gt_classes, max_iou_indices)
            objectness_gt_with_bg = tf.one_hot(
                    tf.cast(class_gt, tf.int32),
                    depth=self.dataset.num_classes+1,)
            objectness_gt = objectness_gt_with_bg[:, 1:]
            #make sure neg anchor's class vector is all zeros
            objectness_gt = tf.where(neg_anchor_mask, tf.zeros_like(objectness_gt), objectness_gt)
            anchors_gt = tf.gather(gt_anchors, max_iou_indices)
            offsets_gt = anchor_bev_encoder.tf_anchor_to_offset(anchors, anchors_gt)
            if add_h:
                label_h = tf.stack([gt_boxes_3d[:, 1], gt_boxes_3d[:, 5]], axis=1)
                gt_anchor_h = tf.gather(label_h, max_iou_indices)
                #shape: (num_anchor,)
                n_anchor = tf.shape(max_iou_indices)[0]
                anchor_h = anchor_bev_encoder.get_default_anchor_h(n_anchor, fmt='tf')
                offsets_h_gt = anchor_bev_encoder.anchor_to_offset_h(anchor_h, gt_anchor_h)
            if add_angle:
                gt_anchor_angle = tf.gather(gt_anchors[:, -1], max_iou_indices)
                offsets_angle_cls_gt = orientation_encoder.tf_orientation_to_angle_cls(gt_anchor_angle)

        #these info is pred at num anchors = feat_h*feat_w
        #should be filtered later by non empty indices
        targets = {
            self.PRED_OBJECTNESS_GT: objectness_gt,
            self.PRED_OFFSETS_GT: offsets_gt,
            self.PRED_OFFSETS_H_GT: offsets_h_gt,
            self.PRED_OFFSETS_ANGLE_CLS_GT: offsets_angle_cls_gt,
            self.POS_ANCHORS_MASK: pos_anchor_mask,
            self.NEG_ANCHORS_MASK: neg_anchor_mask,
        }
 
        return targets


    def get_nms_result(self, pred_results, visualize=False):
        with tf.variable_scope('nms'):
            #pred_objectness = pred_results[self.PRED_OBJECTNESS] 
            pred_anchors = pred_results[self.PRED_ANCHORS]
            pred_objectness_sigmoid = tf.sigmoid(pred_results[self.PRED_OBJECTNESS], name='sigmoid')
            #only one class
            #pred_objectness_sigmoid = tf.reshape(pred_objectness_sigmoid, (-1, 1))
            pred_objectness_sigmoid = tf.reshape(pred_objectness_sigmoid, (-1,))
            pred_offsets = pred_results[self.PRED_OFFSETS] 
            regressed_anchors = anchor_bev_encoder.offset_to_anchor(
                    pred_anchors, pred_offsets)
            #Do score filtering to speed up NMS time..
            
            nms_score_thresh = -0.01
            #shape is (N, 1)
            nms_indices = tf.where(tf.greater_equal(pred_objectness_sigmoid, nms_score_thresh))
            #nms_indices = tf.argsort(pred_objectness_sigmoid, direction='DESCENDING')[:15000]
            #nms_indices = tf.reshape(nms_indices, (-1, 1))
            regressed_anchors = tf.gather_nd(regressed_anchors, nms_indices)
            pred_objectness_sigmoid = tf.gather_nd(pred_objectness_sigmoid, nms_indices)

            #[TODO] rotate nms!!
            # Do NMS on regressed anchors
            device_id = 0
            iou_threshold = self._nms_iou_thresh
            #iou_threshold = 0.1
            #concat anchors and probs
            det_tensor = tf.concat(\
                    [regressed_anchors, tf.reshape(pred_objectness_sigmoid, (-1, 1))],\
                    axis=1)
            keep = tf.py_func(rotate_gpu_nms,
                    inp=[det_tensor, iou_threshold, device_id],
                    Tout=tf.int64)
            keep = tf.reshape(keep, [-1,])
            top_indices = tf.cond(
                tf.greater(tf.shape(keep)[0], self._nms_size),
                true_fn=lambda: tf.slice(keep, [0], [self._nms_size]),
                false_fn=lambda: keep)
            #all_indices = tf.range(0, tf.shape(regressed_anchors)[0])
            #top_indices = all_indices
            top_anchors = tf.gather(regressed_anchors, top_indices)
            top_objectness_sigmoid = tf.gather(pred_objectness_sigmoid,
                                               top_indices)
            anchors_info_list = [top_anchors]
            if pred_results[self.PRED_OFFSETS_H] is not None:
                pred_offsets_h = pred_results[self.PRED_OFFSETS_H]
                offset_h_gt_weight = 1.0 #4.5
                pred_offsets_h /= offset_h_gt_weight
                top_offsets_h = tf.gather(pred_offsets_h, top_indices)
                n_anchor = tf.shape(top_indices)[0]
                anchor_h = anchor_bev_encoder.get_default_anchor_h(n_anchor, fmt='tf')
                top_anchor_h = anchor_bev_encoder.offset_to_anchor_h(anchor_h, top_offsets_h)
                anchors_info_list.append(top_anchor_h)
 
            if pred_results[self.PRED_OFFSETS_ANGLE_CLS] is not None:
                pred_offsets_angle_cls = pred_results[self.PRED_OFFSETS_ANGLE_CLS]
                top_offsets_angle_cls_logits = tf.gather(pred_offsets_angle_cls, top_indices)
                top_anchor_angle_cls = tf.argmax(top_offsets_angle_cls_logits, axis=1)
                top_anchor_angle_cls = tf.reshape(tf.cast(top_anchor_angle_cls, tf.float32), (-1, 1))
                anchors_info_list.append(top_anchor_angle_cls)
            top_anchors_info = tf.concat(anchors_info_list, axis=1)
            top_anchor_scores = tf.reshape(top_objectness_sigmoid, (-1,))
            nms_result = {
                    self.PRED_TOP_INDICES: top_indices,
                    self.PRED_TOP_ANCHORS: top_anchors_info,
                    self.PRED_TOP_OBJECTNESS_SIGMOID: top_objectness_sigmoid,
                    }
            if visualize:
                vis_mask = tf.greater(top_anchor_scores, 0.4)
                vis_top_anchors = tf.boolean_mask(top_anchors, vis_mask)
                vis_top_anchor_scores = tf.boolean_mask(top_anchor_scores, vis_mask)

                with tf.variable_scope('visualize'):
                    top_in_bev = show_box_in_tensor.draw_boxes_with_scores(
                            img_batch=self._bev_input_batches,
                            boxes=vis_top_anchors,
                            scores=vis_top_anchor_scores,
                            method=1)  
                    tf.summary.image('top_pred_boxes_bev', top_in_bev)
                if pred_results[self.PRED_OFFSETS_H] is not None:
                    vis_top_h = tf.boolean_mask(top_anchor_h, vis_mask)
                else:
                    vis_top_h = anchor_bev_encoder.get_default_anchor_h(
                            tf.shape(vis_top_anchors)[0], 'tf')
                if pred_results[self.PRED_OFFSETS_ANGLE_CLS] is not None:
                    vis_top_angle_cls = tf.boolean_mask(top_anchor_angle_cls, vis_mask)
                    vis_top_angle_cls = tf.reshape(vis_top_angle_cls, (-1, ))
                else:
                    vis_top_angle_cls = tf.ones(tf.shape(vis_top_anchors)[0])
                with tf.variable_scope('project_to_img'):
                    top_box_3d = tf.py_func(box_bev_encoder.box_bev_to_box_3d,
                        inp=[vis_top_anchors, self._bev_pixel_size, self._bev_extents, \
                                vis_top_h],\
                        Tout=tf.float32)
                    top_box_img = tf.py_func(box_bev_encoder.boxes_3d_project_to_image,
                            inp=[top_box_3d, self.placeholders[self.PL_CALIB_P2]],
                            Tout=tf.float32) 
                    #pred boxes and image
                    #top_in_rgb = show_box_in_tensor.only_draw_boxes(
                    #        img_batch=self._img_input_batches,
                    #        boxes=top_box_img,
                    #        method=0)  
                    angle_label = vis_top_angle_cls + 1 # zero means bkg. 1 means (0, pi), 2 means (-pi, 0)
                    top_in_rgb = show_box_in_tensor.draw_boxes_with_categories(
                            img_batch=self._img_input_batches,
                            boxes=top_box_img,
                            labels=angle_label,
                            method=0)  
                    

                    tf.summary.image('pred_boxes_img', top_in_rgb)

        return nms_result

    def visualize_positive_anchor(self, anchors, pos_mask, objectness, offsets, offsets_gt, offsets_h=None):
        pos_anchors = tf.boolean_mask(anchors, pos_mask)
        objectness = tf.reshape(objectness, (-1, )) #ONLY ONE CLASS
        probs = tf.sigmoid(objectness, name='sigmoid')
        pos_probs = tf.boolean_mask(probs, pos_mask)
        pos_in_bev = show_box_in_tensor.draw_boxes_with_scores(
                    img_batch=self._bev_input_batches,
                    boxes=pos_anchors,
                    scores=pos_probs,
                    method=1)  
        #tf.summary.image('pos_anchors', pos_in_bev)
        pos_anchor_offsets_gt = tf.boolean_mask(offsets_gt, pos_mask)
        pos_anchor_gt = anchor_bev_encoder.offset_to_anchor(pos_anchors, pos_anchor_offsets_gt)
        gt_in_bev = show_box_in_tensor.only_draw_boxes(
                    img_batch=self._bev_input_batches,
                    boxes=pos_anchor_gt,
                    method=1)  
        tf.summary.image('gt_boxes_bev', gt_in_bev)
        draw_pred = False #True
        if draw_pred:
            pos_anchor_offset = tf.boolean_mask(offsets, pos_mask)
            pos_anchor_pred = anchor_bev_encoder.offset_to_anchor(pos_anchors, pos_anchor_offset)
            pos_pred_in_bev = show_box_in_tensor.draw_boxes_with_scores(
                        img_batch=self._bev_input_batches,
                        boxes=pos_anchor_pred,
                        scores=pos_probs,
                        method=1)  
            tf.summary.image('pred_boxes_bev', pos_pred_in_bev)
            with tf.variable_scope('project_to_img'):
                if offsets_h is None:
                    pos_pred_3d = tf.py_func(box_bev_encoder.box_bev_to_box_3d,
                        inp=[pos_anchor_pred, self._bev_pixel_size, self._bev_extents],
                        Tout=tf.float32)
                else:
                    pos_anchor_offset_h = tf.boolean_mask(offsets_h, pos_mask)
                    n_anchor = tf.shape(pos_anchor_offset_h)[0]
                    anchor_h = anchor_bev_encoder.get_default_anchor_h(n_anchor, fmt='tf')
                    pos_anchor_pred_h = anchor_bev_encoder.offset_to_anchor_h(anchor_h, pos_anchor_offset_h)
                    pos_pred_3d = tf.py_func(box_bev_encoder.box_bev_to_box_3d,
                        inp=[pos_anchor_pred, self._bev_pixel_size, self._bev_extents, \
                                pos_anchor_pred_h],\
                        Tout=tf.float32)
                
                   
                pos_pred_img = tf.py_func(box_bev_encoder.boxes_3d_project_to_image,
                        inp=[pos_pred_3d, self.placeholders[self.PL_CALIB_P2]],
                        Tout=tf.float32) 
                #pred boxes and image
                pos_pred_in_rgb = show_box_in_tensor.only_draw_boxes(
                        img_batch=self._img_input_batches,
                        boxes=pos_pred_img,
                        method=0)  
                tf.summary.image('pred_boxes_img', pos_pred_in_rgb)


