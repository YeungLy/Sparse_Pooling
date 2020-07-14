"""Common functions for evaluating checkpoints.
"""
import datetime
import time
import os
import subprocess
import numpy as np
import cv2
from multiprocessing import Process
import tensorflow as tf

from avod.core import box_3d_encoder
from avod.core import evaluator_utils
from avod.core import summary_utils
from avod.core import trainer_utils

from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core.models.retinanet_model import RetinanetModel
from avod.core import box_bev_encoder 
from avod.core import orientation_encoder
from wavedata.tools.core import calib_utils

tf.logging.set_verbosity(tf.logging.INFO)

KEY_SUM_CLS_LOSS = 'sum_cls_loss'
KEY_SUM_REG_LOSS = 'sum_reg_loss'
KEY_SUM_REG_H_LOSS = 'sum_reg_h_loss'
KEY_SUM_REG_ANGLE_CLS_LOSS = 'sum_reg_angle_cls_loss'
KEY_SUM_TOTAL_LOSS = 'sum_total_loss'
KEY_SUM_CLS_ACC = 'sum_obj_accuracy'
KEY_NUM_VALID_REG_SAMPLES = 'num_valid_reg_samples'


class EvaluatorRetinanet:

    def __init__(self,
                 model,
                 dataset_config,
                 eval_config,
                 skip_evaluated_checkpoints=True,
                 eval_wait_interval=30,
                 do_kitti_native_eval=True):
        """Evaluator class for evaluating model's detection output.

        Args:
            model: An instance of DetectionModel
            dataset_config: Dataset protobuf configuration
            eval_config: Evaluation protobuf configuration
            skip_evaluated_checkpoints: (optional) Enables checking evaluation
                results directory and if the folder names with the checkpoint
                index exists, it 'assumes' that checkpoint has already been
                evaluated and skips that checkpoint.
            eval_wait_interval: (optional) The number of seconds between
                looking for a new checkpoint.
            do_kitti_native_eval: (optional) flag to enable running kitti native
                eval code.
        """

        # Get model configurations
        self.model = model
        self.dataset_config = dataset_config
        self.eval_config = eval_config

        self.model_config = model.model_config
        self.model_name = self.model_config.model_name

        rlosses_keys = [f'refine{i}' for i in range(self.model.refine_stage_num)]
        self.model_losses_keys = ['fcn'] + rlosses_keys
        self.model_has_h_flags = self.model.add_h_flags
        self.model_has_angle_flags = self.model.add_angle_flags

        self.paths_config = self.model_config.paths_config
        self.checkpoint_dir = self.paths_config.checkpoint_dir

        self.skip_evaluated_checkpoints = skip_evaluated_checkpoints
        self.eval_wait_interval = eval_wait_interval

        self.do_kitti_native_eval = do_kitti_native_eval

        # Create a variable tensor to hold the global step
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')

        eval_mode = eval_config.eval_mode
        if eval_mode not in ['val', 'test']:
            raise ValueError('Evaluation mode can only be set to `val`'
                             'or `test`')

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        if self.do_kitti_native_eval:
            if self.eval_config.eval_mode == 'val':
                # Copy kitti native eval code into the predictions folder
                evaluator_utils.copy_kitti_native_code(
                    self.model_config.checkpoint_name)

        allow_gpu_mem_growth = self.eval_config.allow_gpu_mem_growth
        if allow_gpu_mem_growth:
            # GPU memory config
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = allow_gpu_mem_growth
            self._sess = tf.Session(config=config)
        else:
            self._sess = tf.Session()

        # The model should return a dictionary of predictions
        #self._prediction_dict = dict()
        self._prediction_dict = self.model.build()
        if eval_mode == 'val':
            # Setup loss and summary writer in val mode only
            self._loss_dict, self._total_loss = \
                self.model.loss(self._prediction_dict)

            self.summary_writer, self.summary_merged = \
                evaluator_utils.set_up_summary_writer(self.model_config,
                                                      self._sess)
            logdir = self.paths_config.logdir
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            logdir = logdir + '/eval'

            datetime_str = str(datetime.datetime.now())+'sample2'
            self.summary_writer2 = tf.summary.FileWriter(logdir + '/' + datetime_str)


        else:
            self._loss_dict = None
            self._total_loss = None
            self.summary_writer = None
            self.summary_merged = None

        self._saver = tf.train.Saver()

        # Add maximum memory usage summary op
        # This op can only be run on device with gpu
        # so it's skipped on travis
        is_travis = 'TRAVIS' in os.environ
        if not is_travis:
            # tf 1.4
            # tf.summary.scalar('bytes_in_use',
            #                   tf.contrib.memory_stats.BytesInUse())
            tf.summary.scalar('max_bytes',
                              tf.contrib.memory_stats.MaxBytesInUse())

    def run_checkpoint_once(self, checkpoint_to_restore):
        """Evaluates network metrics once over all the validation samples.

        Args:
            checkpoint_to_restore: The directory of the checkpoint to restore.
        """

        self._saver.restore(self._sess, checkpoint_to_restore)

        data_split = self.dataset_config.data_split
        predictions_base_dir = self.paths_config.pred_dir

        num_samples = self.model.dataset.num_samples
        train_val_test = self.model._train_val_test
        print('model: train_val_test: ', train_val_test)

        validation = train_val_test == 'val'

        global_step = trainer_utils.get_global_step(
            self._sess, self.global_step_tensor)

        # Rpn average losses dictionary
        if validation:
            sum_losses = self._create_losses_dict()


        # Make sure the box representation is valid
        predictions_dir = predictions_base_dir + \
                "/final_predictions_and_scores/{}/{}".format(
                data_split, global_step)
        trainer_utils.create_dir(predictions_dir)

        num_valid_samples = 0

        # Keep track of feed_dict and inference time
        total_feed_dict_time = []
        total_inference_time = []

        # Run through a single epoch
        current_epoch = self.model.dataset.epochs_completed

        #run_metadata = tf.RunMetadata()
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        while current_epoch == self.model.dataset.epochs_completed:
            # Keep track of feed_dict speed
            start_time = time.time()
            #feed_dict = self.model.create_feed_dict(sample_index=sample_index)
            feed_dict = self.model.create_feed_dict()
            feed_dict_time = time.time() - start_time

            # Get sample name from model
            sample_name = self.model.sample_info['sample_name']
            stereo_calib = calib_utils.read_calibration(self.model.dataset.calib_dir,
                                                           int(sample_name))
            stereo_calib_p2 = stereo_calib.p2
            
            output_file_path = predictions_dir + \
                "/{}.txt".format(sample_name)

            num_valid_samples += 1
            #if num_valid_samples > 1:
            #    break
            print("Step {}: {} / {}, Inference on sample {}".format(
                global_step, num_valid_samples, num_samples,
                sample_name))


            # Do predictions, loss calculations, and summaries

            if validation:
                if self.summary_merged is not None:
                    predictions, eval_losses, eval_total_loss, summary_out = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss,
                                        self.summary_merged],
                                       feed_dict=feed_dict)
 
                    if num_valid_samples == 2 and num_samples == 2:
                        self.summary_writer2.add_summary(summary_out, global_step)
                    else:
                        self.summary_writer.add_summary(summary_out, global_step)

                else:
                    print('start inference without smry:')
                    predictions, eval_losses, eval_total_loss = \
                        self._sess.run([self._prediction_dict,
                                        self._loss_dict,
                                        self._total_loss],
                                       feed_dict=feed_dict)
                                       #options=run_options,
                                       #run_metadata=run_metadata)
                    #self.summary_writer.add_run_metadata(run_metadata, \
                    #        'step {} sp:{}'.format(global_step/1000, int(sample_name)))


                self._update_losses(eval_losses,
                                    eval_total_loss,
                                    sum_losses,
                                    global_step)
                # Save predictions

                print('save predictions')
                predictions_and_scores = \
                    self.get_predicted_boxes_3d_and_scores(predictions,
                                                            stereo_calib_p2)
                np.savetxt(output_file_path, predictions_and_scores, fmt='%.5f')

                # Calculate accuracies
                #Unnecessary because there is only one class.. object class without bkg class..
                self.get_cls_accuracy(predictions,
                                      sum_losses,
                                      global_step)
                print("Step {}: Total time {} s".format(
                    global_step, time.time() - start_time))

            else:
                # Test mode --> train_val_test == 'test'
                inference_start_time = time.time()
                # Don't calculate loss or run summaries for test
                predictions = self._sess.run(self._prediction_dict,
                                             feed_dict=feed_dict)
                inference_time = time.time() - inference_start_time

                # Add times to list
                total_feed_dict_time.append(feed_dict_time)
                total_inference_time.append(inference_time)

                predictions_and_scores = \
                    self.get_predicted_boxes_3d_and_scores(predictions,
                                                            stereo_calib_p2)
                np.savetxt(file_path, predictions_and_scores, fmt='%.5f')

        # end while current_epoch == model.dataset.epochs_completed:

        if validation:
                # Kitti native evaluation, do this during validation
                # and when running Avod model.
                # Store predictions in kitti format
            self.save_prediction_losses_results(sum_losses, num_valid_samples, \
                    global_step, predictions_base_dir)
            if self.do_kitti_native_eval:
                pass
                #self.run_kitti_native_eval(global_step)

        else:
            # Test mode --> train_val_test == 'test'
            evaluator_utils.print_inference_time_statistics(
                total_feed_dict_time, total_inference_time)

        print("Step {}: Finished evaluation, results saved to {}".format(
            global_step, predictions_dir))

    def run_latest_checkpoints(self):
        """Evaluation function for evaluating all the existing checkpoints.
        This function just runs through all the existing checkpoints.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Load the latest checkpoints available
        trainer_utils.load_checkpoints(self.checkpoint_dir,
                                       self._saver)

        num_checkpoints = len(self._saver.last_checkpoints)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config)

        ckpt_indices = np.asarray(self.eval_config.ckpt_indices)
        if ckpt_indices is not None:
            if ckpt_indices[0] == -1:
                # Restore the most recent checkpoint
                ckpt_idx = num_checkpoints - 1
                ckpt_indices = [ckpt_idx]
            for ckpt_idx in ckpt_indices:
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                self.run_checkpoint_once(checkpoint_to_restore)

        else:
            last_checkpoint_id = -1
            number_of_evaluations = 0
            # go through all existing checkpoints
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = evaluator_utils.strip_checkpoint_id(
                    checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                already_evaluated = ckpt_id in already_evaluated_ckpts
                if already_evaluated or ckpt_id <= last_checkpoint_id:
                    number_of_evaluations = max((ckpt_idx + 1,
                                                 number_of_evaluations))
                    continue

                self.run_checkpoint_once(checkpoint_to_restore)
                number_of_evaluations += 1

                # Save the id of the latest evaluated checkpoint
                last_checkpoint_id = ckpt_id

    def repeated_checkpoint_run(self):
        """Periodically evaluates the checkpoints inside the `checkpoint_dir`.

        This function evaluates all the existing checkpoints as they are being
        generated. If there are none, it sleeps until new checkpoints become
        available. Since there is no synchronization guarantee for the trainer
        and evaluator, at each iteration it reloads all the checkpoints and
        searches for the last checkpoint to continue from. This is meant to be
        called in parallel to the trainer to evaluate the models regularly.

        Raises:
            ValueError: if model.checkpoint_dir doesn't have at least one
                element.
        """

        if not os.path.exists(self.checkpoint_dir):
            raise ValueError('{} must have at least one checkpoint entry.'
                             .format(self.checkpoint_dir))

        # Copy kitti native eval code into the predictions folder
        if self.do_kitti_native_eval:
            evaluator_utils.copy_kitti_native_code(
                self.model_config.checkpoint_name)

        if self.skip_evaluated_checkpoints:
            already_evaluated_ckpts = self.get_evaluated_ckpts(
                self.model_config)
        else:
            already_evaluated_ckpts = []
        tf.logging.info(
            'Starting evaluation at ' +
            time.strftime(
                '%Y-%m-%d-%H:%M:%S',
                time.gmtime()))

        last_checkpoint_id = -1
        number_of_evaluations = 0
        #Dont have to add summary(for model inference at each sample) at repeated evaluation.. 
        #only care avg loss at each ckpt step.
        #self.summary_merged = None
        evaluated_ckpts = [ckpt for ckpt in already_evaluated_ckpts]
        while True:
            # Load current checkpoints available
            trainer_utils.load_checkpoints(self.checkpoint_dir,
                                           self._saver)
            num_checkpoints = len(self._saver.last_checkpoints)
            no_newckpts = True
            evaluated_ckpts.sort()
            start = time.time()
            for ckpt_idx in range(num_checkpoints):
                checkpoint_to_restore = \
                    self._saver.last_checkpoints[ckpt_idx]
                ckpt_id = evaluator_utils.strip_checkpoint_id(
                    checkpoint_to_restore)

                # Check if checkpoint has been evaluated already
                if ckpt_id == 0 or ckpt_id in evaluated_ckpts:
                    continue
                else:
                    no_newckpts = False
                print('evaluated ckpts: ', evaluated_ckpts)
                print('processing ckpt id: ', ckpt_id)
                self.run_checkpoint_once(checkpoint_to_restore)
                evaluated_ckpts.append(ckpt_id)
            time_to_next_eval = start + self.eval_wait_interval - time.time()
            if no_newckpts:
                tf.logging.info('No new checkpoints found in %s.'
                                'Will try again in %d seconds',
                                self.checkpoint_dir,
                                self.eval_wait_interval)
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)

    def _update_losses(self,
                           eval_losses,
                           eval_total_loss,
                           sum_losses,
                           global_step):
        """Helper function to calculate the evaluation average losses.

        Args:
            eval_losses: A dictionary of network's output
            eval_total_loss: A scalar loss of rpn total loss.
            sum_losses: A dictionary containing all the average
                losses.
            global_step: Global step at which the metrics are computed.
        """

        is_valid_reg_sample = True
        for sk in self.model_losses_keys:
            #check all stage's reg loss to verify if valid reg sample
            reg_loss = eval_losses[sk][RetinanetModel.LOSS_RETINANET_REGRESSION]
            if reg_loss <= 0.0:
                is_valid_reg_sample = False
        losses_str = ''
        for sk, has_h, has_ang in zip(\
                self.model_losses_keys, self.model_has_h_flags,\
                self.model_has_angle_flags):
            stage_losses = eval_losses[sk]
            cls_loss = stage_losses[RetinanetModel.LOSS_RETINANET_OBJECTNESS]
            sum_losses[sk][KEY_SUM_CLS_LOSS] += cls_loss
            losses_str += f'{sk} cls: {cls_loss}, '
            if is_valid_reg_sample:
                reg_loss = stage_losses[RetinanetModel.LOSS_RETINANET_REGRESSION]
                sum_losses[sk][KEY_SUM_REG_LOSS] += reg_loss
                losses_str += f'reg: {reg_loss}, '
                if has_h:
                    reg_h_loss = stage_losses[RetinanetModel.LOSS_RETINANET_H]
                    sum_losses[sk][KEY_SUM_REG_H_LOSS] += reg_h_loss
                    losses_str += f'h: {reg_h_loss}, '
                if has_ang:
                    reg_angle_cls_loss = stage_losses[RetinanetModel.LOSS_RETINANET_ANGLE_CLS]
                    sum_losses[sk][KEY_SUM_REG_ANGLE_CLS_LOSS] += reg_angle_cls_loss
                    losses_str += f'angle: {reg_angle_cls_loss}, '

            losses_str += '\n'

        #print(f"Step {global_step}: Eval Loss: {losses_str}")
        sum_losses[KEY_SUM_TOTAL_LOSS] += eval_total_loss
        if is_valid_reg_sample:
            sum_losses[KEY_NUM_VALID_REG_SAMPLES] += 1

    def save_prediction_losses_results(self,
                                       sum_losses,
                                       num_valid_samples,
                                       global_step,
                                       predictions_base_dir):
        """Helper function to save the AVOD loss evaluation results.

        Args:
            eval_avod_losses: A dictionary containing the loss sums
            num_valid_samples: An int, number of valid evaluated samples
                i.e. samples with valid ground-truth.
            global_step: Global step at which the metrics are computed.
            predictions_base_dir: Base directory for storing the results.
            box_rep: A string, the format of the 3D bounding box
                one of 'box_3d', 'box_8c' etc.
        """
        num_valid_reg_samples = max(1, sum_losses[KEY_NUM_VALID_REG_SAMPLES])
        avg_losses_key_str = ''
        avg_losses_print_str = ''
        avg_losses_data = []

        for sk, has_h, has_ang in zip(\
                self.model_losses_keys, self.model_has_h_flags,\
                self.model_has_angle_flags):
            stage_losses = sum_losses[sk]
            sum_cls_loss = stage_losses[KEY_SUM_CLS_LOSS]
            avg_cls_loss = sum_cls_loss / num_valid_samples
            # Write summaries
                #f'{sk}_losses/classification/cls',
            summary_utils.add_scalar_summary(
                f'{sk}_loss/retinanet_losses/cls/norm/val',
                avg_cls_loss,
                self.summary_writer, global_step)

            sum_reg_loss = stage_losses[KEY_SUM_REG_LOSS]
            avg_reg_loss = sum_reg_loss / num_valid_reg_samples
                #f'{sk}_losses/regression/reg',
            summary_utils.add_scalar_summary(
                f'{sk}_loss/retinanet_losses/reg/norm/val',
                avg_reg_loss,
                self.summary_writer, global_step)
            avg_losses_print_str += f'[{sk}]cls:{avg_cls_loss}, reg:{avg_reg_loss}, '
            avg_losses_key_str += f'{sk}_cls {sk}_reg '
            avg_losses_data.extend([avg_cls_loss, avg_reg_loss])
            if KEY_SUM_REG_H_LOSS in stage_losses:
                sum_h_loss = stage_losses[KEY_SUM_REG_H_LOSS]
                avg_h_loss = sum_h_loss / num_valid_reg_samples 
                    #f'{sk}_losses/regression/h',
                summary_utils.add_scalar_summary(
                    f'{sk}_loss/retinanet_losses/reg/norm/h_reg',
                    avg_h_loss,
                    self.summary_writer, global_step)
                avg_losses_print_str += f'h:{avg_h_loss}, '
                avg_losses_key_str += f'{sk}_h '
                avg_losses_data.append(avg_h_loss)
            if KEY_SUM_REG_ANGLE_CLS_LOSS in stage_losses:
                sum_angle_cls_loss = stage_losses[KEY_SUM_REG_ANGLE_CLS_LOSS]
                avg_angle_cls_loss = sum_angle_cls_loss / num_valid_reg_samples 
                    #f'{sk}_losses/regression/angle_cls',
                summary_utils.add_scalar_summary(
                    f'{sk}_loss/retinanet_losses/reg/norm/angle_cls',
                    avg_angle_cls_loss,
                    self.summary_writer, global_step)
                avg_losses_print_str += f'angle:{avg_angle_cls_loss}, '
                avg_losses_key_str += f'{sk}_angle '
                avg_losses_data.append(avg_angle_cls_loss)

        sum_cls_acc = sum_losses[KEY_SUM_CLS_ACC]
        avg_cls_acc = sum_cls_acc / num_valid_samples
        summary_utils.add_scalar_summary(
            f'output/cls_accuracy',
            avg_cls_acc,
            self.summary_writer, global_step)
        
        self.summary_writer.flush()

        sum_total_loss = sum_losses[KEY_SUM_TOTAL_LOSS]
        avg_total_loss = sum_total_loss / num_valid_samples
        avg_losses_print_str += f'\ntotal:{avg_total_loss}'
        avg_losses_key_str += 'total '
        avg_losses_data.append(avg_total_loss)

        # Append to end of file
        avg_loss_file_dir = predictions_base_dir + '/avg_losses/'\
                + self.dataset_config.data_split  
        if not os.path.exists(avg_loss_file_dir):
            os.makedirs(avg_loss_file_dir)
        avg_loss_file_path = avg_loss_file_dir +'/avg_losses.csv'
        if not os.path.exists(avg_loss_file_path):
            with open(avg_loss_file_path, 'w') as f:
                f.write(f'Step {avg_losses_key_str}\n')

        save = True
        if save:
            print(f"Step {global_step}: Average Losses: \n{avg_losses_print_str}")

            with open(avg_loss_file_path, 'ba') as fp:
                avg_losses_data = np.array(avg_losses_data)
                np.savetxt(fp,
                       [np.hstack(
                        [global_step, avg_losses_data]
                        )],
                       fmt='%d'+', %.5f'*(len(avg_losses_data)))


    def _create_losses_dict(self):
        """Returns a dictionary of the losses sum for averaging.
        """
        sum_losses = dict()

        # Initialize Rpn average losses
        for stage_key, has_h, has_ang in zip(\
                self.model_losses_keys, self.model_has_h_flags,\
                self.model_has_angle_flags):
            stage_losses = dict() 
            stage_losses[KEY_SUM_CLS_LOSS] = 0
            stage_losses[KEY_SUM_REG_LOSS] = 0
            if has_h:
                stage_losses[KEY_SUM_REG_H_LOSS] = 0
            if has_ang:
                stage_losses[KEY_SUM_REG_ANGLE_CLS_LOSS] = 0 
            sum_losses[stage_key] = stage_losses

        sum_losses[KEY_SUM_TOTAL_LOSS] = 0
        sum_losses[KEY_SUM_CLS_ACC] = 0
        sum_losses[KEY_NUM_VALID_REG_SAMPLES] = 0

        return sum_losses

    def get_evaluated_ckpts(self,
                            model_config,
                            ):
        """Finds the evaluated checkpoints.

        Examines the evaluation average losses file to find the already
        evaluated checkpoints.

        Args:
            model_config: Model protobuf configuration

        Returns:
            already_evaluated_ckpts: A list of checkpoint indices, or an
                empty list if no evaluated indices are found.
        """

        already_evaluated_ckpts = []

        # check for previously evaluated checkpoints
        # regardless of model, we are always evaluating rpn, but we do
        # this check based on model in case the evaluator got interrupted
        # and only saved results for one model
        paths_config = model_config.paths_config

        predictions_base_dir = paths_config.pred_dir
        avg_loss_file_dir = predictions_base_dir + '/avg_losses/'\
                + self.dataset_config.data_split  
        avg_loss_file_path = avg_loss_file_dir +'/avg_losses.csv'

        if os.path.exists(avg_loss_file_path):
            avg_losses = np.loadtxt(avg_loss_file_path, delimiter=',',  skiprows=1)
            print(avg_losses)
            if avg_losses.ndim == 1:
                # one entry
                already_evaluated_ckpts = np.asarray(
                    [avg_losses[0]], np.int32)
            else:
                already_evaluated_ckpts = np.asarray(avg_losses[:, 0],
                                                     np.int32)

        return already_evaluated_ckpts

    def get_cls_accuracy(self,
                         predictions,
                         sum_losses,
                         global_step):
        """Updates the calculated accuracies for rpn and avod losses.

        Args:
            predictions: A dictionary containing the model outputs.
            eval_avod_losses: A dictionary containing all the avod averaged
                losses.
            eval_rpn_losses: A dictionary containing all the rpn averaged
                losses.
            global_step: Current global step that is being evaluated.
        """
        final = self.model.STAGE_KEYS[-1] 
        predictions = predictions[final]
        cls_pred = predictions[RetinanetModel.PRED_OBJECTNESS]
        cls_gt = predictions[RetinanetModel.PRED_OBJECTNESS_GT]
        cls_gt = cls_gt.astype(np.int32)
        cls_gt_logits = np.eye(2)[cls_gt]
        cls_pred_logits = np.stack([1-cls_pred, cls_pred], axis=1)
        cls_accuracy = self.calculate_cls_accuracy(cls_pred_logits,
                                                  cls_gt_logits)

        # get this from the key
        sum_cls_accuracy = sum_losses[KEY_SUM_CLS_ACC]
        sum_cls_accuracy += cls_accuracy
        sum_losses.update({KEY_SUM_CLS_ACC:
                                sum_cls_accuracy})
        print("Step {}: RetinaNet Classification Accuracy: {}".format(
            global_step, cls_accuracy))

    def calculate_cls_accuracy(self, cls_pred, cls_gt):
        """Calculates accuracy of predicted objectness/classification wrt to
        the labels

        Args:
            cls_pred: A numpy array containing the predicted
            objectness/classification values in the form (mini_batches, 2)
            cls_gt: A numpy array containing the ground truth
            objectness/classification values in the form (mini_batches, 2)

        Returns:
            accuracy: A scalar value representing the accuracy
        """
        correct_prediction = np.equal(np.argmax(cls_pred, 1),
                                      np.argmax(cls_gt, 1))
        accuracy = np.mean(correct_prediction)
        return accuracy

    def get_predicted_boxes_3d_and_scores(self, predictions,
                                               stereo_calib_p2):
        """Returns the predictions and scores stacked for saving to file.

        Args:
            predictions: A dictionary containing the model outputs.

        Returns:
            predictions_and_scores: A numpy array of shape
                (number_of_predicted_boxes, 9), containing the final prediction
                boxes, orientations, scores, and types.
        """

        if not self.model.do_nms_at_gpu:
            predictions = self.run_nms_cpu(predictions)
        final_pred_anchors = predictions[
                RetinanetModel.PRED_TOP_ANCHORS]
        final_pred_sigmoid = predictions[
                RetinanetModel.PRED_TOP_OBJECTNESS_SIGMOID]
        anchors = final_pred_anchors[:, :5]
        col_h_lo = 5
        col_h_hi = col_h_lo
        if self.model.add_h:
            col_h_hi = col_h_lo + 2
            h = final_pred_anchors[:, col_h_lo:col_h_hi]
        else:
            h = None
        if self.model.add_angle:
            col_angle_cls = col_h_hi
            angle_cls = final_pred_anchors[:, col_angle_cls]
            angle_val = anchors[:, -1]
            angle = orientation_encoder.angle_clsval_to_orientation(angle_cls, angle_val)
            anchors[:, -1] = angle
        bev_shape = [self.model_config.input_config.bev_dims_h,\
                     self.model_config.input_config.bev_dims_w]
        area_extents = self.dataset_config.kitti_utils_config.area_extents
        bev_extents = (area_extents[0:2], area_extents[4:6])
        final_pred_boxes_3d = box_bev_encoder.box_bev_to_box_3d(\
                anchors, bev_shape, bev_extents, h)
 
        # Append score and class index (object type)
        #shape is (number_of_top_anchors, num_class[no background]) 
       
        #ONLY ONE CLASS. BESIDES, nms is done for each class. 0 means 1th object type.
        final_pred_types = np.zeros_like(final_pred_sigmoid, dtype=np.int32)
        final_pred_scores = final_pred_sigmoid
           
        predictions_and_scores = np.column_stack(
            [final_pred_boxes_3d,
             final_pred_scores,
             final_pred_types])

        return predictions_and_scores

    def run_kitti_native_eval(self, global_step):
        """Calls the kitti native C++ evaluation code.

        It first saves the predictions in kitti format. It then creates two
        child processes to run the evaluation code. The native evaluation
        hard-codes the IoU threshold inside the code, so hence its called
        twice for each IoU separately.

        Args:
            global_step: Global step of the current checkpoint to be evaluated.
        """

        # Kitti native evaluation, do this during validation
        # and when running Avod model.
        # Store predictions in kitti format
        evaluator_utils.save_predictions_in_kitti_format(
            self.model,
            self.model_config.checkpoint_name,
            self.dataset_config.data_split,
            self.eval_config.kitti_score_threshold,
            global_step)

        checkpoint_name = self.model_config.checkpoint_name
        kitti_score_threshold = self.eval_config.kitti_score_threshold

        # Create a separate processes to run the native evaluation
        #native_eval_proc = Process(
        #    target=evaluator_utils.run_kitti_native_script, args=(
        #        checkpoint_name, kitti_score_threshold, global_step))

        eval_script_dir = self.paths_config.pred_dir #predictions_base_dir
        native_eval_proc = Process(
                target=run_my_kitti_native_script,
                args=(eval_script_dir, 
                    self.dataset_config.data_split,
                    global_step,
                    '07'))
        native_eval_proc_05_iou = Process(
                target=run_my_kitti_native_script,
                args=(eval_script_dir, 
                    self.dataset_config.data_split,
                    global_step,
                    '05')) #iou05=True
        #native_eval_proc_05_iou = Process(
        #    target=evaluator_utils.run_kitti_native_script_with_05_iou,
        #    args=(checkpoint_name, kitti_score_threshold, global_step))
        # Don't call join on this cuz we do not want to block
        # this will cause one zombie process - should be fixed later.
        native_eval_proc.start()
        native_eval_proc_05_iou.start()

    def run_nms_cpu(self, predictions):
        pred_anchors = predictions[\
                RetinanetModel.PRED_TOP_ANCHORS]
        pred_sigmoid = predictions[\
                RetinanetModel.PRED_TOP_OBJECTNESS_SIGMOID]

        #do NMS
        pred_boxes = pred_anchors[:, :5]
        boxes = pred_boxes
        boxes[:, -1] *= (180 / np.pi) #convert rad to degree for openCV
        scores = np.reshape(pred_sigmoid, (-1,))
        max_output_size = 100
        iou_threshold = 0.3

        keep = []

        order = scores.argsort()[::-1]
        num = boxes.shape[0]

        suppressed = np.zeros((num), dtype=np.int)
        start = time.time()
        for _i in range(num):
            if len(keep) >= max_output_size:
                break

            i = order[_i]
            if suppressed[i] == 1:
                continue
            keep.append(i)
            r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
            area_r1 = boxes[i, 2] * boxes[i, 3]
            for _j in range(_i + 1, num):
                j = order[_j]
                if suppressed[i] == 1:
                    continue
                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

                if inter >= iou_threshold:
                    suppressed[j] = 1

        #return np.array(keep, np.int64)
        print('cost: ', time.time() -start)
        keep = keep[:max_output_size]
        print(len(boxes), len(keep), keep[:5])
        predictions[RetinanetModel.PRED_TOP_ANCHORS] = pred_anchors[keep]
        predictions[RetinanetModel.PRED_TOP_OBJECTNESS_SIGMOID] = pred_sigmoid[keep]
        return predictions
    #TODO : TEST THIS FUNCTION!!!!!


def run_my_kitti_native_script(eval_script_dir, data_split, global_step, iou='05'):

    if iou == '05': 
        make_script = eval_script_dir + \
            '/kitti_native_eval/run_05iou.sh'
    elif iou == '07':
        make_script = eval_script_dir + \
            '/kitti_native_eval/run.sh'
    else:
        raise ValueError('Wrong IoU threshold.')
    script_folder = os.path.dirname(make_script)
    subprocess.call([make_script, script_folder, 
                     str(global_step),
                     str(data_split),])
 



