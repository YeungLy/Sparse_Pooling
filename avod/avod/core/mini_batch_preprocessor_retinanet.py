import numpy as np
import os

from PIL import Image

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation

from avod.core import box_3d_encoder, anchor_projector
from avod.core import box_3d_projector, box_bev_encoder
from avod.core import anchor_encoder
from avod.core import anchor_bev_encoder
from avod.core import orientation_encoder
from avod.core import anchor_filter

from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.core.anchor_generators import grid_anchor_bev_generator

from avod.core.feature_extractors.bev_resnet_fpn import BevResnetFpn

class MiniBatchPreprocessorForRetinanet(object):
    def __init__(self,
                 dataset,
                 mini_batch_dir,
                 anchor_params,
                 bev_shape,
                 neg_iou_range,
                 pos_iou_range):
        """Preprocesses anchors and saves info to files for RPN training

        Args:
            dataset: Dataset object
            mini_batch_dir: directory to save the info
            anchor_strides: anchor strides for generating anchors (per class)
            density_threshold: minimum number of points required to keep an
                anchor
            neg_iou_3d_range: 3D iou range for an anchor to be negative
            pos_iou_3d_range: 3D iou range for an anchor to be positive
        """

        self._dataset = dataset
        self.mini_batch_utils = self._dataset.kitti_utils.mini_batch_utils

        self._mini_batch_dir = mini_batch_dir

        self._area_extents = self._dataset.kitti_utils.area_extents
        self._anchor_params = anchor_params
        self._bev_shape = bev_shape
        self._density_threshold = 1

        self._negative_iou_range = neg_iou_range
        self._positive_iou_range = pos_iou_range

    def _calculate_anchors_info(self,
                                all_anchor_boxes_bev,
                                empty_anchor_filter,
                                gt_labels):
        """Calculates the list of anchor information in the format:
            N x 8 [max_gt_2d_iou_r, max_gt_2d_iou_h, (6 x offsets), class_index]
                max_gt_out - highest 2D iou with any ground truth box, using [anchor_r vs gt_r] or [anchor_h vs gt_h]
                offsets - encoded offsets [dx, dy, d_dimx, d_dimy, d_angle, angle_face_class_index, (-180,0) or (0,180)]
                class_index - the anchor's class as an index
                    (e.g. 0 or 1, for "Background" or "Car")

        Args:
            all_anchor_boxes_3d: list of anchors in box_3d format
                N x [xc, yc, w, h, angle]
            empty_anchor_filter: boolean mask of which anchors are non empty
            gt_labels: list of Object Label data format containing ground truth
                labels to generate positives/negatives from.

        Returns:
            list of anchor info
        """
        # Check for ground truth objects
        if len(gt_labels) == 0:
            raise Warning("No valid ground truth label to generate anchors.")

        kitti_utils = self._dataset.kitti_utils

        # Filter empty anchors
        anchor_indices = np.where(empty_anchor_filter)[0]
        anchors = all_anchor_boxes_bev[empty_anchor_filter]

        # Convert anchor_boxes_3d to anchor format
        #anchors = box_bev_encoder.box_bev_to_anchor(anchor_boxes_bev)

        # Convert gt to boxes_3d -> anchors -> iou format
        gt_boxes_3d = np.asarray(
            [box_3d_encoder.object_label_to_box_3d(gt_obj)
             for gt_obj in gt_labels])
        gt_anchors_norm, _ = box_3d_projector.project_to_bev_box(gt_boxes_3d, self._area_extents[[0, 2]])
        #bev_image_size = kitti_utils.area_extents / kitti_utils.voxel_size
        bev_map_h, bev_map_w = self._bev_shape
        #(N, 5) , (5, ) coorespondence element multiplification
        gt_anchors = np.multiply(gt_anchors_norm, np.array([bev_map_w, bev_map_h, bev_map_w, bev_map_h, 1]))
        iou_type = self.mini_batch_utils.retinanet_iou_type
        if iou_type == '2d_rotate':
            # Convert anchors to 2d iou format
            anchors_for_2d_iou_r = anchors 
            gt_boxes_for_2d_iou_r = gt_anchors
        elif iou_type == '2d':
            # Convert anchors to 3d iou format for calculation
            anchors_for_2d_iou_h = box_bev_encoder.box_bev_to_iou_h_format(anchors)
            anchors_for_2d_iou_h = anchors_for_2d_iou_h.astype(np.int32)
            gt_boxes_for_2d_iou_h = box_bev_encoder.box_bev_to_iou_h_format(gt_anchors)
            gt_boxes_for_2d_iou_h = gt_boxes_for_2d_iou_h.astype(np.int32)
        else:
            raise ValueError('Invalid retinanet iou_type {}', iou_type)
        # Initialize sample and offset lists
        num_anchors = len(anchors)
        all_info = np.zeros((num_anchors,
                             self.mini_batch_utils.col_length))
        # Update anchor indices
        all_info[:, self.mini_batch_utils.col_anchor_indices] = anchor_indices

        # For each of the labels, generate samples
        for gt_idx in range(len(gt_labels)):

            gt_obj = gt_labels[gt_idx]
            gt_box_3d = box_3d_encoder.object_label_to_box_3d(gt_obj)

            # Get 2D or 3D IoU for every anchor
            if self.mini_batch_utils.retinanet_iou_type == '2d':
                gt_box_for_2d_iou_h = gt_boxes_for_2d_iou_h[gt_idx]
                ious = evaluation.two_d_iou(gt_box_for_2d_iou_h,
                                            anchors_for_2d_iou_h)
            elif self.mini_batch_utils.retinanet_iou_type == '2d_rotate':
                gt_box_for_2d_iou_r = gt_boxes_for_2d_iou_r[gt_idx]
                ious = evaluation.two_d_rotate_iou(gt_box_for_2d_iou_r,
                                                    anchors_for_2d_iou_r)

            # Only update indices with a higher iou than before
            update_indices = np.greater(
                ious, all_info[:, self.mini_batch_utils.col_ious])

            # Get ious to update
            ious_to_update = ious[update_indices]

            # Calculate offsets, use 3D iou to get highest iou
            anchors_to_update = anchors[update_indices]
            facing_obj_head = gt_obj.ry >= 0  #camera facing object's head. 
            gt_anchor = gt_anchors[gt_idx]
            #turns (-pi, pi) to (-pi, 0) for gt_anchor's angle
            if facing_obj_head:
                gt_anchor[-1] -= np.pi
            offsets_boxes = anchor_bev_encoder.anchor_to_offset(anchors_to_update,
                                                      gt_anchor)
            gt_anchor_pred = anchor_bev_encoder.offset_to_anchor(anchors_to_update, 
                                                      offsets_boxes)
            #y axis 3d value
            n_anchor = offsets_boxes.shape[0]
            anchor_h = anchor_bev_encoder.get_default_anchor_h(n_anchor, 'np')
            gt_h = [gt_obj.t[1], gt_obj.h]
            offsets_h = anchor_bev_encoder.anchor_to_offset_h(anchor_h, gt_h) 
            gt_anchors_angle = np.zeros_like(offsets_boxes[:, 0], dtype=np.int) + gt_obj.ry
            offsets_angle_cls = orientation_encoder.orientation_to_angle_cls(gt_anchors_angle)  
            offsets = np.hstack([offsets_boxes, offsets_h, offsets_angle_cls[:, np.newaxis]])
            # Convert gt type to index
            class_idx = kitti_utils.class_str_to_index(gt_obj.type)
            # Update anchors info (indices already updated)
            # [index, iou, (offsets), class_index]
            all_info[update_indices,
                     self.mini_batch_utils.col_ious] = ious_to_update
            all_info[update_indices,
                     self.mini_batch_utils.col_offsets_lo:
                     self.mini_batch_utils.col_offsets_hi] = offsets
            all_info[update_indices,
                     self.mini_batch_utils.col_class_idx] = class_idx
            debug = False #True
            if debug:
                print(f'gt obj:{gt_box_3d}, gt anchor bev: {gt_anchor}')
                print(f'anchors_to_update: {anchors_to_update[:1]}')
                print(f'update at all_info: \n{all_info[update_indices][:1]}')
                print(f'gt_from_anchor_offsets:\n{gt_anchor_pred[:1]}')

        return all_info

    def preprocess(self, indices):
        """Preprocesses anchor info and saves info to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        anchor_params = self._anchor_params

        dataset = self._dataset
        dataset_utils = self._dataset.kitti_utils
        classes_name = dataset.classes_name

        anchor_strides = anchor_params['anchor_strides']
        # Make folder if it doesn't exist yet
        output_dir = self.mini_batch_utils.get_file_path(classes_name,
                                                         anchor_strides,
                                                         sample_name=None)
        os.makedirs(output_dir, exist_ok=True)

        # Get clusters for class
        #all_clusters_sizes, _ = dataset.get_cluster_info()

        anchor_generator = grid_anchor_bev_generator.GridAnchorBevGenerator()
        #anchor_type = self._dataset.kitti_utils.anchor_type

        # Load indices of data_split
        all_samples = dataset.sample_list

        if indices is None:
            indices = np.arange(len(all_samples))

        #indices = indices[:10]
        num_samples = len(indices)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            # Get image name for given cluster
            sample_name = all_samples[sample_idx].name
            img_idx = int(sample_name)

            # Check for existing files and skip to the next
            if self._check_for_existing(classes_name, anchor_strides,
                                        sample_name):
                print("{} / {}: Sample already preprocessed".format(
                    sample_idx + 1, num_samples, sample_name))
                #continue

            # Get ground truth and filter based on difficulty
            ground_truth_list = obj_utils.read_labels(dataset.label_dir,
                                                      img_idx)

            # Filter objects to dataset classes
            filtered_gt_list = dataset_utils.filter_labels(ground_truth_list)
            filtered_gt_list = np.asarray(filtered_gt_list)

            # Filtering by class has no valid ground truth, skip this image
            if len(filtered_gt_list) == 0:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                          sample_idx + 1, num_samples,
                          classes_name, sample_name))

                # Output an empty file and move on to the next image.
                #comment out for DEBUG
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            # Get ground plane
            ground_plane = obj_utils.get_road_plane(img_idx,
                                                    dataset.planes_dir)

            image = Image.open(dataset.get_rgb_image_path(sample_name))
            image_shape = [image.size[1], image.size[0]]

            # List for merging all anchors
            all_level_anchor_boxes_bev = anchor_generator.generate(\
                    image_shapes=anchor_params['image_shapes'],
                    anchor_base_sizes=anchor_params['anchor_base_sizes'],
                    anchor_strides=anchor_params['anchor_strides'],
                    anchor_ratios=anchor_params['anchor_ratios'],
                    anchor_scales=anchor_params['anchor_scales'],
                    anchor_init_ry_type=anchor_params['anchor_init_ry_type'])
            #concate all levels anchors
            #commentt out for DEBUG
            all_anchor_boxes_bev = np.concatenate(all_level_anchor_boxes_bev)
            #all_anchor_boxes_bev = all_level_anchor_boxes_bev[-1]

            # Filter empty anchors (whose pts num < density_threshold)
            # prepare for anchors_3d which dont need ry.
            anchors_bev = all_anchor_boxes_bev.copy()
            if anchor_params['anchor_init_ry_type'] == -90:
                anchors_bev[:, [2, 3]] = anchors_bev[:, [3,2]] 
            anchors_3d = box_bev_encoder.box_bev_to_anchor_3d(anchors_bev, \
                    bev_shape=self._bev_shape, \
                    bev_extents=self._dataset.kitti_utils.area_extents[[0, 2]])
            #print(anchors_3d)
            image = Image.open(dataset.get_rgb_image_path(sample_name))
            image_shape = [image.size[1], image.size[0]]
            # Generate sliced 2D voxel grid for filtering
            vx_grid_2d = dataset_utils.create_sliced_voxel_grid_2d(
                sample_name,
                source=dataset.bev_source,
                image_shape=image_shape)
            empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_3d, vx_grid_2d, self._density_threshold)
            print(f'Non empty anchor: {np.sum(empty_anchor_filter)} / {len(all_anchor_boxes_bev)}, \
                    sample_name: {sample_name}')

            #empty_anchor_filter = np.ones(all_anchor_boxes_bev.shape[0], dtype=bool)

            # Calculate anchor info
            anchors_info = self._calculate_anchors_info(
                all_anchor_boxes_bev, empty_anchor_filter, filtered_gt_list)
            n_invalid = np.sum(np.isnan(anchors_info))
            if n_invalid > 0:
                raise ValueError('Invalid value occur at anchors_info: nan, sample: ', sample_name)
#
            anchor_ious = anchors_info[:, self.mini_batch_utils.col_ious]

            valid_iou_indices = np.where(anchor_ious > 0.0)[0]

            print("{} / {}:"
                  "{:>6} anchors, "
                  "{:>6} iou > 0.0, "
                  "for {:>3} {}(s) for sample {}".format(
                      sample_idx + 1, num_samples,
                      len(anchors_info),
                      len(valid_iou_indices),
                      len(filtered_gt_list), classes_name, sample_name
                  ))

            # Save anchors info
            #comment out for DEBUG
            self._save_to_file(classes_name, anchor_strides,
                               sample_name, anchors_info)

    def _check_for_existing(self, classes_name, anchor_strides, sample_name):
        """
        Checks if a mini batch file exists already

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): sample name from dataset, e.g. '000123'

        Returns:
            True if the anchors info file already exists
        """

        file_name = self.mini_batch_utils.get_file_path(classes_name,
                                                        anchor_strides,
                                                        sample_name)
        if os.path.exists(file_name):
            return True

        return False

    def _save_to_file(self, classes_name, anchor_strides, sample_name,
                      anchors_info=np.array([])):
        """
        Saves the anchors info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            anchors_info: ndarray of anchor info of shape (N, 8)
                N x [index, iou, (6 x offsets), class_index], defaults to
                an empty array
        """

        file_name = self.mini_batch_utils.get_file_path(classes_name,
                                                        anchor_strides,
                                                        sample_name)

        # Save to npy file
        anchors_info = np.asarray(anchors_info, dtype=np.float32)
        np.save(file_name, anchors_info)
