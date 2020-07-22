import tensorflow as tf
from avod.utils.sparse_pool_utils import sparse_pool_layer
slim = tf.contrib.slim

from avod.builders import feature_extractor_builder


class FusionResnetFpn:

    def __init__(self, 
            bev_extractor_config, 
            img_extractor_config):

        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                bev_extractor_config)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                img_extractor_config)
        

    def build(self, inputs, is_training, fusion_params, scope='fusion_resnet_fpn'):
        bev_inputs, img_inputs = inputs
        bev_feature_dict = self._bev_feature_extractor.build_resnet(bev_inputs, is_training)
        img_feature_dict = self._img_feature_extractor.build_resnet(img_inputs, is_training)
        #before fusion
        with tf.variable_scope('build_fusion'):
            bev_fpn_channel = self._bev_feature_extractor.FPN_CHANNEL
            bev_weight_decay = self._bev_feature_extractor.config.weight_decay
            bev_P5 = slim.conv2d(bev_feature_dict['C5'],
                    num_outputs=bev_fpn_channel,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_regularizer=slim.l2_regularizer(bev_weight_decay),
                    scope='bev_build_P5'
                    )
            img_fpn_channel = self._img_feature_extractor.FPN_CHANNEL
            img_weight_decay = self._img_feature_extractor.config.weight_decay
            img_P5 = slim.conv2d(img_feature_dict['C5'],
                    num_outputs=img_fpn_channel,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_regularizer=slim.l2_regularizer(img_weight_decay),
                    scope='img_build_P5'
                    )

            M_tf = fusion_params['M_tf']
            img_index_flip = fusion_params['img_index_flip']
            feature_depths = [img_fpn_channel, bev_fpn_channel] 
            bev_P5_fused, img_P5_fused = sparse_pool_layer(\
                    [bev_P5, img_P5],\
                    feature_depths,\
                    M_tf,\
                    img_index_flip = img_index_flip, \
                    bv_index = None, \
                    training=is_training)
            print('WZN: Successfully created Fusion at resnet-P5 \n')

        bev_feature_dict['P5'] = bev_P5_fused
        img_feature_dict['P5'] = img_P5_fused
        bev_pyramid_dict = self._bev_feature_extractor.build_fpn(bev_feature_dict, fuse_at_p5=True)
        img_pyramid_dict = self._img_feature_extractor.build_fpn(img_feature_dict, fuse_at_p5=True)

        return bev_pyramid_dict, img_pyramid_dict


