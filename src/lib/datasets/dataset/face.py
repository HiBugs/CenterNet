#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/2/5
@Description:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
import pycocotools.coco as coco
import torch.utils.data as data
from pycocotools.cocoeval import COCOeval

Num_Key_Point = 5


class Face(data.Dataset):
    num_classes = 1
    num_joints = Num_Key_Point
    default_resolution = [512, 512]
    mean = np.array([0.5, 0.5, 0.5],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[0, 1], [3, 4]]

    def __init__(self, opt, split):
        super(Face, self).__init__()

        # self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
        #               [4, 6], [3, 5], [5, 6],
        #               [5, 7], [7, 9], [6, 8], [8, 10],
        #               [6, 12], [5, 11], [11, 12],
        #               [12, 14], [14, 16], [11, 13], [13, 15]]
        # self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.data_dir = os.path.join(opt.data_dir, 'WiderFace')
        self.img_dir = os.path.join(self.data_dir, 'WIDER_{}/images'.format(split))

        self.annot_path = os.path.join(
            self.data_dir, 'WIDER_{}'.format(split),
            'keypoints_{}.json'.format(split))

        # if split == 'test':
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'WIDER_test',
        #         'keypoints_{}.json').format(split)
        # else:
        #     self.annot_path = os.path.join(
        #         self.data_dir, 'WIDER_{}'.format(split),
        #         'keypoints_{}.json'.format(split))

        self.max_objs = 100
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing WiderFace {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:5 + Num_Key_Point * 2], dtype=np.float32).reshape(-1, 2),
                        np.ones((Num_Key_Point, 1), dtype=np.float32)], axis=1).reshape(Num_Key_Point * 3).tolist()
                    keypoints = list(map(self._to_float, keypoints))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score)),
                        "keypoints": keypoints
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_coco_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def save_widerface_results(self, results, img_path, save_dir):

        dir_name = os.path.join(save_dir, img_path.split("/")[-2])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_prefix = img_path.split("/")[-1].split(".")[0]
        file_suffix = '.txt'
        file_name = file_prefix + file_suffix

        all_bboxes = results[1]
        num_boxes = len(all_bboxes)
        with open(os.path.join(dir_name, file_name), 'w') as f:
            f.write('{:s}\n'.format(file_prefix))
            f.write('{:d}\n'.format(num_boxes))
            for line in all_bboxes:
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.3f}\n'.
                        format(line[0], line[1], line[2] - line[0], line[3] - line[1], line[4]))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_coco_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        # coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
