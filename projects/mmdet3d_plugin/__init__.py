# numpy >= 1.24 removed the deprecated builtin aliases that this plugin and
# mmdet3d v0.17.1 still use (np.bool, np.int, ...). Restore them.
import numpy as _np
for _name, _type in (('bool', bool), ('int', int), ('float', float),
                     ('long', int), ('object', object), ('str', str)):
    if not hasattr(_np, _name):
        setattr(_np, _name, _type)

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.backbones.vovnet import VoVNet
from .models.utils import *
from .models.opt.adamw import AdamW2
from .VAD import *
