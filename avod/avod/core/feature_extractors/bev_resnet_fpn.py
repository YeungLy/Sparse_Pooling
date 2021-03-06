import tensorflow as tf
from avod.core.feature_extractors import bev_feature_extractor

slim = tf.contrib.slim

class BevResnetFpn(bev_feature_extractor.BevFeatureExtractor):
    DATA_FORMAT = "NHWC"
    DEBUG = False
    debug_dict = {}
    BOTTLENECK_NUM_DICT = {
        'resnet50_v1b': [3, 4, 6, 3],
        'resnet101_v1b': [3, 4, 23, 3],
        'resnet152_v1b': [3, 8, 36, 3],
        'resnet50_v1d': [3, 4, 6, 3],
        'resnet101_v1d': [3, 4, 23, 3],
        'resnet152_v1d': [3, 8, 36, 3]
    }

    BASE_CHANNELS_DICT = {
        'resnet50_v1b': [64, 128, 256, 512],
        'resnet101_v1b': [64, 128, 256, 512],
        'resnet152_v1b': [64, 128, 256, 512],
        'resnet50_v1d': [64, 128, 256, 512],
        'resnet101_v1d': [64, 128, 256, 512],
        'resnet152_v1d': [64, 128, 256, 512]
    }
    FREEZE_BLOCKS = [True, False, False, False, False]  # for gluoncv backbone
    FPN_CHANNEL = 256

    def resnet_arg_scope(self, freeze_norm, is_training=True, weight_decay=0.0001,
                     batch_norm_decay=0.9, batch_norm_epsilon=1e-5, batch_norm_scale=True):

        batch_norm_params = {
            'is_training': False, 'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
            'trainable': False,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'data_format': self.DATA_FORMAT
        }
        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                trainable=is_training,
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc

    def stem_7x7(self, net, scope="C1"):

        with tf.variable_scope(scope):
            net = tf.pad(net, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])  # pad for data
            net = slim.conv2d(net, num_outputs=64, kernel_size=[7, 7], stride=2,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope="conv0")
            if self.DEBUG:
                self.debug_dict['conv_7x7_bn_relu'] = tf.transpose(net, [0, 3, 1, 2])  # NHWC --> NCHW
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=self.DATA_FORMAT)
            return net


    def stem_stack_3x3(self, net, input_channel=32, scope="C1"):
        with tf.variable_scope(scope):
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=2,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv0')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel, kernel_size=[3, 3], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv1')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, num_outputs=input_channel*2, kernel_size=[3, 3], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv2')
            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding="VALID", data_format=self.DATA_FORMAT)
            return net


    def bottleneck_v1b(self, input_x, base_channel, scope, stride=1, projection=False, avg_down=True):
        '''
        for bottleneck_v1b: reduce spatial dim in conv_3x3 with stride 2.
        '''
        with tf.variable_scope(scope):
            if self.DEBUG:
                self.debug_dict[input_x.op.name] = tf.transpose(input_x, [0, 3, 1, 2])
            net = slim.conv2d(input_x, num_outputs=base_channel, kernel_size=[1, 1], stride=1, padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT, scope='conv0')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

            net = tf.pad(net, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])

            net = slim.conv2d(net, num_outputs=base_channel, kernel_size=[3, 3], stride=stride,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              scope='conv1')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            net = slim.conv2d(net, num_outputs=base_channel * 4, kernel_size=[1, 1], stride=1,
                              padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                              activation_fn=None, scope='conv2')
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            # Note that : gamma in the last conv should be init with 0.
            # But we just reload params from mxnet, so don't specific batch norm initializer
            if projection:

                if avg_down:  # design for resnet_v1d
                    '''
                    In GluonCV, padding is "ceil mode". Here we use "SAME" to replace it, which may cause Erros.
                    And the erro will grow with depth of resnet. e.g. res101 erro > res50 erro
                    '''
                    shortcut = slim.avg_pool2d(input_x, kernel_size=[stride, stride], stride=stride, padding="SAME",
                                               data_format=self.DATA_FORMAT)
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

                    shortcut = slim.conv2d(shortcut, num_outputs=base_channel*4, kernel_size=[1, 1],
                                           stride=1, padding="VALID", biases_initializer=None, data_format=self.DATA_FORMAT,
                                           activation_fn=None,
                                           scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
                    # shortcut should have batch norm.
                else:
                    shortcut = slim.conv2d(input_x, num_outputs=base_channel * 4, kernel_size=[1, 1],
                                           stride=stride, padding="VALID", biases_initializer=None, activation_fn=None,
                                           data_format=self.DATA_FORMAT,
                                           scope='shortcut')
                    if self.DEBUG:
                        self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])
            else:
                shortcut = tf.identity(input_x, name='shortcut/Identity')
                if self.DEBUG:
                    self.debug_dict[shortcut.op.name] = tf.transpose(shortcut, [0, 3, 1, 2])

            net = net + shortcut
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            net = tf.nn.relu(net)
            if self.DEBUG:
                self.debug_dict[net.op.name] = tf.transpose(net, [0, 3, 1, 2])
            return net


    def make_block(self, net, base_channel, bottleneck_nums, scope, avg_down=True, spatial_downsample=False):
        with tf.variable_scope(scope):
            first_stride = 2 if spatial_downsample else 1

            net = self.bottleneck_v1b(input_x=net, base_channel=base_channel,
                                 scope='bottleneck_0',
                                 stride=first_stride, avg_down=avg_down, projection=True)
            for i in range(1, bottleneck_nums):
                net = self.bottleneck_v1b(input_x=net, base_channel=base_channel, 
                                     scope="bottleneck_%d" % i,
                                     stride=1, avg_down=avg_down, projection=False)
            return net


    def get_resnet_v1_b_base(self, input_x, freeze_norm, scope, bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                        freeze=[True, False, False, False, False], is_training=True):

        assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
        assert len(freeze) == len(bottleneck_nums) +1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
        feature_dict = {}
        with tf.variable_scope(scope):
            with slim.arg_scope(self.resnet_arg_scope(is_training=(not freeze[0]) and is_training,
                                                 freeze_norm=freeze_norm)):
                net = self.stem_7x7(net=input_x, scope="C1")
                feature_dict["C1"] = net
            for i in range(2, len(bottleneck_nums)+2):
                spatial_downsample = False if i == 2 else True
                with slim.arg_scope(self.resnet_arg_scope(is_training=(not freeze[i-1]) and is_training,
                                                     freeze_norm=freeze_norm)):
                    net = self.make_block(net=net, base_channel=base_channels[i-2],
                                     bottleneck_nums=bottleneck_nums[i-2],
                                     scope="C%d" % i,
                                     avg_down=False, spatial_downsample=spatial_downsample)
                    feature_dict["C%d" % i] = net

        return net, feature_dict


    def get_resnet_v1_d_base(self, input_x, freeze_norm, scope, bottleneck_nums=[3, 4, 6, 3], base_channels=[64, 128, 256, 512],
                        freeze=[True, False, False, False, False], is_training=True):

        assert len(bottleneck_nums) == len(base_channels), "bottleneck num should same as base_channels size"
        assert len(freeze) == len(bottleneck_nums) + 1, "should satisfy:: len(freeze) == len(bottleneck_nums) + 1"
        feature_dict = {}
        with tf.variable_scope(scope):
            with slim.arg_scope(self.resnet_arg_scope(is_training=((not freeze[0]) and is_training),
                                                 freeze_norm=freeze_norm)):
                net = self.stem_stack_3x3(net=input_x, input_channel=32, scope="C1")
                feature_dict["C1"] = net
                #print ("finish C1")
            for i in range(2, len(bottleneck_nums)+2):
                spatial_downsample = False if i == 2 else True  # do not downsample in C2
                #print("freeze: {}".format(int(freeze[i-1])))
                with slim.arg_scope(self.resnet_arg_scope(is_training=((not freeze[i-1]) and is_training),
                                                     freeze_norm=freeze_norm)):
                    net = self.make_block(net=net, base_channel=base_channels[i-2],
                                     bottleneck_nums=bottleneck_nums[i-2],
                                     scope="C%d" % i,
                                     avg_down=True, spatial_downsample=spatial_downsample)
                    feature_dict["C%d" % i] = net

        return net, feature_dict

    def fusion_two_layer(self, C_i, P_j, scope):
        '''
        i = j+1
        :param C_i: shape is [1, h, w, c]
        :param P_j: shape is [1, h/2, w/2, 256]
        :return:
        P_i
        '''
        with tf.variable_scope(scope):
            level_name = scope.split('_')[1]

            h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
            upsample_p = tf.image.resize_bilinear(P_j,
                                                  size=[h, w],
                                                  name='up_sample_'+level_name)
            reduce_dim_c = slim.conv2d(C_i,
                                       num_outputs=256,
                                       kernel_size=[1, 1], stride=1,
                                       scope='reduce_dim_'+level_name)

            add_f = 0.5*upsample_p + 0.5*reduce_dim_c
            return add_f

    def build_resnet(self, 
            inputs, 
            is_training, 
            scope='bev_resnet'):
        resnet_name = self.config.resnet_name
        if resnet_name.endswith('b'):
            get_resnet_fn = self.get_resnet_v1_b_base
        elif resnet_name.endswith('d'):
            get_resnet_fn = self.get_resnet_v1_d_base
        else:
            raise ValueError("resnet_name erro....")
        scope += resnet_name[6:]
        print('Building bev feature extractor: ', scope)

        _, feature_dict = get_resnet_fn(input_x=inputs, scope=scope,
                                        bottleneck_nums=self.BOTTLENECK_NUM_DICT[resnet_name],
                                        base_channels=self.BASE_CHANNELS_DICT[resnet_name],
                                        is_training=is_training, freeze_norm=True,
                                        freeze=self.FREEZE_BLOCKS)

        return feature_dict

    def build_fpn(self, feature_dict, fuse_at_p5=False):

        resnet_config = self.config
        pyramid_levels = resnet_config.pyramid_levels
        use_p5_at_p6 = resnet_config.use_p5_at_p6
        use_relu_at_fusion = resnet_config.use_relu_at_fusion
        weight_decay = resnet_config.weight_decay
        pyramid_dict = {}
        with tf.variable_scope('bev_build_feature_pyramid'):
        #FPN part
            with slim.arg_scope([slim.conv2d], 
                    weights_regularizer=slim.l2_regularizer(weight_decay), 
                    activation_fn=None, 
                    normalizer_fn=None):

                if not fuse_at_p5:
                    P5 = slim.conv2d(feature_dict['C5'],
                                 num_outputs=self.FPN_CHANNEL,
                                 kernel_size=[1, 1],
                                 stride=1, scope='build_P5')

                    pyramid_dict['P5'] = P5

                else:
                    if 'P5' not in feature_dict:
                        raise ValueError('P5 should be fused and saved at feature dict.')
                    else:
                        pyramid_dict['P5'] = feature_dict['P5']

                #we can fuse img and bev here. maybe high-level has more semantic.
                #img to bev.

                l_top, l_down = int(pyramid_levels[-1][-1]), int(pyramid_levels[0][-1])
                for level in range(4, l_down - 1, -1):  # build [P4, P3]
                    pyramid_dict[f'P{level}'] = self.fusion_two_layer(
                            C_i=feature_dict[f'C{level}'], 
                            P_j=pyramid_dict[f'P{level+1}'],
                            scope='build_P%d' % level)
                #for level in range(5, l_down - 1, -1):
                for level in range(l_top, l_down - 1, -1):
                    pyramid_dict[f'P{level}'] = slim.conv2d(pyramid_dict[f'P{level}'],
                            num_outputs=self.FPN_CHANNEL, 
                            kernel_size=[3, 3], 
                            padding="SAME",
                            stride=1, 
                            scope=f'fuse_P{level}',
                            activation_fn=tf.nn.relu if use_relu_at_fusion else None)
                if 'P6' in pyramid_levels:
                    p6_input = pyramid_dict['P5'] if use_p5_at_p6 else feature_dict['C5']
                    p6 = slim.conv2d(p6_input,
                                 num_outputs=self.FPN_CHANNEL, 
                                 kernel_size=[3, 3], padding="SAME",
                                 stride=2, 
                                 scope='P6_conv')
                    pyramid_dict['P6'] = p6
                    if 'P7' in pyramid_levels:
                        p7 = tf.nn.relu(p6, name='P6_relu')
                        p7 = slim.conv2d(p7,
                                 num_outputs=self.FPN_CHANNEL, 
                                 kernel_size=[3, 3], padding="SAME",
                                 stride=2, 
                                 scope='P7_conv')
                        pyramid_dict['P7'] = p7

        return pyramid_dict

 
    def build(self,
              inputs,
              is_training,
              scope='bev_resnet'):


        feature_dict = self.build_resnet(inputs, is_training, scope)
        pyramid_dict = self.build_fpn(feature_dict)

        return pyramid_dict


