#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/2/9
@Description:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import os

import cv2
import numpy as np
import torch
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from logger import Logger
from opts import opts
from progress.bar import Bar
from utils.utils import AverageMeter


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # opt.dataset = 'face'
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for ind in range(num_iters):
        img_id = dataset.images[ind]

        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        if opt.task == 'ddd':
            ret = detector.run(img_path, img_info['calib'])
        else:
            ret = detector.run(img_path)

        # results[img_id] = ret['results']
        dataset.save_widerface_results(ret['results'], img_path, "eval_wider")

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    # dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().parse()
    test(opt)
