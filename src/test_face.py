#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/2/4
@Description:
"""
import sys
CENTERNET_PATH = "/nfs/home/zjb/workspace/CenterNet/src/lib/"
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = "../exp/multi_pose/widerface_dla34/model_best.pth"
TASK = 'multi_pose' # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

label_names = ['face']
Num_Key_Point = 5

img = "../images/10_People_Marching_People_Marching_2_962.jpg"
ret = detector.run(img)['results']


