# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .coco_wholebody import COCO_WHOLEBODYDataset as coco_wholebody
from .halpe_136 import HALPE_136_Dataset as halpe_136
from .halpe_26 import HALPE_26_Dataset as halpe_26