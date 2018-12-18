# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import os.path as osp

from config import cfg


def _solver_proto_from_template():
    with open(osp.join(cfg.EXP_PATH, 'solver.tpl')) as f:
        s = f.read()
        s = s.replace('<dataset>', cfg.DATASET)
        s = s.replace('<exp>', cfg.EXP)
        s = s.replace('<infix>', cfg.INFIX)
        s = s.replace('<base_lr>', str(cfg.BASE_LR))

    with open(cfg.SL_PATH, 'w') as f:
        f.write(s)


def _train_proto_from_template():
    with open(osp.join(cfg.EXP_PATH, 'train.tpl')) as f:
        s = f.read()
        s = s.replace('<num_bbox_pred_regout>',
                      str(2 * len(cfg.TRAIN.ANCHOR_SCALES)))
        s = s.replace('<inner_margin>',
                      str(cfg.TRAIN.CLIP_INNER_MARGIN))
        s = s.replace('<outer_margin>',
                      str(cfg.TRAIN.CLIP_OUTER_MARGIN))

    with open(cfg.TR_PATH, 'w') as f:
        f.write(s)


def prototxt_from_template():
    _solver_proto_from_template()
    _train_proto_from_template()
