from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import json_tricks as json
import pickle
import numpy as np

from ..dataset.JointsDataset import JointsDataset
from ..nms.nms import oks_nms
from ..nms.nms import soft_oks_nms

logger = logging.getLogger(__name__)

class HALPE_136_Dataset(JointsDataset):
    pass