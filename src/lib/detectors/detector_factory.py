from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .ddd import DddDetector
from .exdet import ExdetDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
    'exdet': ExdetDetector,
    'ddd': DddDetector,
    'ctdet': CtdetDetector,
    'multi_pose': MultiPoseDetector,
}
