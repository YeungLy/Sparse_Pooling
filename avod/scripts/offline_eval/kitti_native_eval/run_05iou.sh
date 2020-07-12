#!/bin/bash
cd $1
gt_dir=~/Dataset/kitti/object/training/label_2
step=$2
split=$3
det_dir=../kitti_predictions_3d/$split/0.1/$step
#det_dir='./0.1/$step'
result_file=results_05iou.txt
echo $step' at split: '$split >> $result_file
echo 'evaluate at '$(date +"%Y-%m-%d %H:%M:%S") >> $result_file
./evaluate_object_3d_offline_05_iou $gt_dir $det_dir | tee -a $result_file
echo done >> $result_file

