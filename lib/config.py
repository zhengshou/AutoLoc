# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import numpy as np
import os.path as osp

from easydict import EasyDict as edict


__C = edict()

cfg = __C


############################################################
# MISC

# Debug option for layers
__C.DEBUG = False

# Currently automatically set 20 for 'TH14' and 100 for 'AN'
__C.NUM_CLASSES = 20
__C.NUM_FEAT = 2048

__C.BASE_LR = 1e-3
__C.MAX_ITER = 25

# Optional infix for prototxts and snapshot path
#   - exp/<dataset>/<exp>/proto/{solver,train}[_<infix>_].prototxt
#   - exp/<dataset>/<exp>/snapshot/autoloc[_<infix>_]
__C.INFIX = ''

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
# Experiments directory
__C.EXP_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'exp'))
# Snapshot directory
__C.SNAPSHOT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'snapshot'))

# Default GPU device id
__C.GPU_ID = 0

# Whether or not save model when training, remember to open it
# up when model needs to be tested.
__C.SAVE_MODEL = False


############################################################
# Feature post-processing options

__C.FEAT = edict()

# 'mul' or 'mask'
__C.FEAT.MODE = 'mul'
# 'sigmoid' or 'softmax' or 'relu'
__C.FEAT.ACTIVATION = 'sigmoid'
__C.FEAT.NORM = False
__C.FEAT.THRESH = None


############################################################
# Visualization options

__C.VIZ = edict()

# This arg would be overriden by `FEAT` configs.
__C.VIZ.FOLDER_NAME = 'viz'

# Iteration interval to plot optimization plot for training.
__C.VIZ.PLOT_ITVL = 5

# Whether or not fix bar's width and scratching figure length when plotting.
__C.VIZ.FIX_WIDTH = True

# Sub-dataset for visualization groundtruth through `tools/viz_gt.py`
__C.VIZ.STAGE = 'val'

__C.VIZ.PLOT_PR_CURVE = False
__C.VIZ.PLOT_PR_NCOLS = 5


############################################################
# Data spec

__C.DSPEC = edict()

__C.DSPEC.TH14_GT = 'data/TH14/' + \
    '{stage}_gt_dump.py2.pc'
__C.DSPEC.AN_GT = 'data/AN/' + \
    '{stage}_gt_dump.indexified.pc'

__C.DSPEC.TH14_META = 'data/TH14/' + \
    '{stage}_meta_det_only.pc'
__C.DSPEC.AN_META = 'data/AN/' + \
    '{stage}_meta_v2_includedv3.indexified.pc'

__C.DSPEC.WINSTEP = 15


############################################################
# Layer spec

__C.LSPEC = edict()

# Use 'sgl' or 'mlt' position optimization.
__C.LSPEC.OIC_POS_OPT = 'sgl'

# Use 'org' or 'ref' score for anchor selection.
__C.LSPEC.OIC_ANC_OPT = 'org'


############################################################
# Training options

__C.TRAIN = edict()

# Data file for training
#   - data/<dataset>/_<data_file>_
__C.TRAIN.DATA_FILE = 'default'
# Sub-dataset for training
__C.TRAIN.STAGE = 'val'
# Optional folder name for visualization
#   - exp/<dataset>/<exp>/_<viz_folder>_

__C.TRAIN.COL_LABEL = 1
__C.TRAIN.COL_FEAT = 2
__C.TRAIN.COL_HEATMAP = 3
__C.TRAIN.COL_ATT = 4

# Legacy per from `py-faster-rcnn` for CNN kernel scaling back to
# original size. Since now we are doing temporal convolution, we set
# this parameter equal to 1.
__C.TRAIN.FEAT_STRIDE = 1

# Anchor scales ranged in 2**(<lo>, <hi>)
__C.TRAIN.ANCHOR_SCALES = (2 ** np.arange(4, 6)).tolist()

__C.TRAIN.OUTER_INFLATE_RATIO = .25
__C.TRAIN.OUTER_MIN = 1

__C.TRAIN.FEAT_SCALE = 1

__C.TRAIN.CLIP_INNER_MARGIN = 1
__C.TRAIN.CLIP_OUTER_MARGIN = 2

__C.TRAIN.FG_THRESH = .5
# Upper tolerance of OIC loss for optimization
__C.TRAIN.OIC_LOSS_THRESH = .3
# Only used when not choose <_adaptive_nms> mode
__C.TRAIN.NMS_BBOX_THRESH = .2

# Use top-k class when selecting classes according to avg heatmap scores.
# When <OIC_cls_topk> equals to -1, use ground truth labels.
__C.TRAIN.OIC_CLS_TOPK = -1


############################################################
# Testing options

__C.TEST = edict()

# Data file for training
#   - data/<dataset>/_<data_file>_
__C.TEST.DATA_FILE = 'default'
# Sub-dataset for testing
__C.TEST.STAGE = 'test'
# Optional folder name for visualization
#   - exp/<dataset>/<exp>/_<viz_folder>_

__C.TEST.COL_LABEL = 1
__C.TEST.COL_FEAT = 2
__C.TEST.COL_HEATMAP = 3
__C.TEST.COL_ATT = 4

# Legacy per from `py-faster-rcnn` for CNN kernel scaling back to
# original size. Since now we are doing temporal convolution, we set
# this parameter equal to 1.
__C.TEST.FEAT_STRIDE = 1

# Anchor scales ranged in 2**(<lo>, <hi>)
__C.TEST.ANCHOR_SCALES = (2 ** np.arange(4, 6)).tolist()

__C.TEST.OUTER_INFLATE_RATIO = .25
__C.TEST.OUTER_MIN = 1

__C.TEST.FEAT_SCALE = 1

__C.TEST.CLIP_INNER_MARGIN = 1
__C.TEST.CLIP_OUTER_MARGIN = 2

__C.TEST.FG_THRESH = .5
# Upper tolerance of OIC loss for optimization
__C.TEST.OIC_LOSS_THRESH = .3
# Only used when not choose <_adaptive_nms> mode
__C.TEST.NMS_BBOX_THRESH = .2

# Use top-k class when selecting classes according to avg heatmap scores.
# When <OIC_cls_topk> equals to -1, use ground truth labels.
__C.TEST.OIC_CLS_TOPK = 2


############################################################
# Evaluation options

__C.EVAL = edict()

# To enumerate <_tiou_thresh_> - <_nms_shift_> for evaluation.
# Note it will results in len(<_tiou_thresh>) models to save.
__C.EVAL.TRAIN_ADAPTIVE_NMS = False
__C.EVAL.TEST_ADAPTIVE_NMS = False

__C.EVAL.TRAIN_ADAPTIVE_NMS_SHIFT = 0.
__C.EVAL.TEST_ADAPTIVE_NMS_SHIFT = 0.

__C.EVAL.TIOU_THRESH = np.linspace(0.1, 0.7, 7).tolist()

# Precision with predictions of top[x]% OIC_score.
__C.EVAL.PREC_AT_TOPX = .05

# Whether or not include ap results in final table
__C.EVAL.TBL_INCLUDE_CLS = False


############################################################
# Configuration helpers from `py-faster-rcnn`
def _merge_a_into_b(a, b):
    '''
    Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    '''
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # Types must match, too
        old_type = type(b[k])
        if old_type is not type(v) and b[k] is not None:
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # Recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                raise ValueError('Error under config key: {} with {} vs. {}'. \
                                 format(k, a[k], b[k]))
        else:
            b[k] = v


def cfg_from_file(filename):
    '''Load a config file and merge it into the default options.'''
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
