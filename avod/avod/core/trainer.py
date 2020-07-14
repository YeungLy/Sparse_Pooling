"""Detection model trainer.

This file provides a generic training method to train a
DetectionModel.
"""
import datetime
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import avod
from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core import summary_utils
from tensorflow.python import debug as tfdbg

slim = tf.contrib.slim


def train(model, train_config):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
            training i.e. loading RPN weights onto AVOD model.
    """

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    global_step_tensor = tf.Variable(
        0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = \
        train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    paths_config = model_config.paths_config
    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint_dir = paths_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + \
        model_config.checkpoint_name

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build()


    #WZN: for debug only
    #img_input_debug = model._rpn_model._img_preprocessed
    #bev_input_debug = model._rpn_model._bev_preprocessed
    #bev_pooled_debug = model._rpn_model.bev_input_pooled
    #img_pooled_debug = model._rpn_model.img_input_pooled
    #import numpy as np 
    #import matplotlib.pyplot as plt
    

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images
    summary_bev_images = train_config.summary_bev_images

    ##############################
    # Setup loss
    ##############################

    losses_dict, total_loss = model.loss(prediction_dict)

    # Optimizer
    training_optimizer = optimizer_builder.build(
        train_config.optimizer,
        global_summaries,
        global_step_tensor)

    # Create the train op
    with tf.variable_scope('train_op'):
        train_op = slim.learning.create_train_op(
            total_loss,
            training_optimizer,
            clip_gradient_norm=1.0,
            global_step=global_step_tensor)

    # Save checkpoints regularly.
    saver = tf.train.Saver(max_to_keep=max_checkpoints,
                       pad_step_number=True)

    # Add the result of the train_op to the summary
    tf.summary.scalar("training_loss", train_op)

    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis
    is_travis = 'TRAVIS' in os.environ
    if not is_travis:
        # tf.summary.scalar('bytes_in_use',
        #                   tf.contrib.memory_stats.BytesInUse())
        tf.summary.scalar('max_bytes',
                          tf.contrib.memory_stats.MaxBytesInUse())

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    sm_name = [sm.name for sm in summaries]
    summary_merged = summary_utils.summaries_to_keep(
        summaries,
        global_summaries,
        histograms=summary_histograms,
        input_imgs=summary_img_images,
        input_bevs=summary_bev_images
    )
    # Create init op
    init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            )
    init = tf.global_variables_initializer()

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    if allow_gpu_mem_growth:
        # GPU memory config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_mem_growth
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    logdir = logdir + '/train'
    train_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                         sess.graph)

    init_from_pretrained_backbone = False
    # Continue from last saved checkpoint
    if not train_config.overwrite_checkpoints:
        trainer_utils.load_checkpoints(checkpoint_dir,
                                       saver)
        if len(saver.last_checkpoints) > 0:
            checkpoint_to_restore = saver.last_checkpoints[-1]
            print('restore from last checkpoint {}'.format(\
                    checkpoint_to_restore))
            saver.restore(sess, checkpoint_to_restore)
        else:
            # Initialize the variables
            sess.run(init)
            if train_config.load_pretrained_backbone:
                init_from_pretrained_backbone = True
    else:
        # Initialize the variables
        sess.run(init)
        if train_config.load_pretrained_backbone:
            init_from_pretrained_backbone = True

    if init_from_pretrained_backbone:
        restore_variables, backbone_name = model.restore_pretrained_backbone_variables()
        pretrained_backbone_ckpt_path = avod.root_dir() + \
                '/data/pretrained_backbone/resnet/{}.ckpt'.format(backbone_name)
        #only for pretrained backbone
        restorer = tf.train.Saver(var_list=restore_variables)
        print('restore from pretrianed backbone: {}'.format(\
                pretrained_backbone_ckpt_path))
        restorer.restore(sess, pretrained_backbone_ckpt_path)
 
    # Open tf debug
    #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)

    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    print('Starting from step {} / {}'.format(
        global_step, max_iterations))


    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # Main Training Loop
    last_time = time.time()
    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.train.global_step(sess,
                                               global_step_tensor)

            saver.save(sess,
                       save_path=checkpoint_path,
                       global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        feed_dict = model.create_feed_dict()

        #WZN: only run for debug 
        #bev_p,img_p = sess.run([bev_pooled_debug,img_pooled_debug],feed_dict)
        #bev_p = np.squeeze(bev_p)
        #img_p = np.squeeze(img_p)
        #import pdb
        #pdb.set_trace()

        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            train_op_loss, summary_out = sess.run(
                [train_op, summary_merged], feed_dict=feed_dict)

            #train_op_loss, summary_out = sess.run(
            #    [train_op, summary_merged], feed_dict=feed_dict, \
            #    options=run_options, run_metadata=run_metadata)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)

        else:
            # Run the train op only
            sess.run(train_op, feed_dict)

            #sess.run(train_op, feed_dict,\
            #    options=run_options, run_metadata=run_metadata)

        #Timeline profile
        #tl = timeline.Timeline(run_metadata.step_stats)
        #ctf = tl.generate_chrome_trace_format()
        #with open('timeline.json', 'w') as wd:
        #    wd.write(ctf)



    # Close the summary writers
    train_writer.close()
