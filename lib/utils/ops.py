# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import os
import re
import multiprocessing

import numpy as np
import pandas as pd
import os.path as osp
import scipy.stats as st

from config import cfg


############################################################
# Global Mappings
LABEL_IDX = [7,9,12,21,22,23,24,26,31,33,36,40,45,51,68,79,85,92,93,97]
TO20_IDX = { k: v for k, v in zip(LABEL_IDX, range(1, 21)) }
TO101_IDX = { k: v for k, v in zip(range(1, 21), LABEL_IDX) }

# Global lock for parallel training. (See tools.train_prll_lock)
prll_lock = multiprocessing.Lock()


############################################################
# Anchor Generation
def _whctrs(anchor):
    w = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    return w, x_ctr


def _mkanchors(ws, x_ctr):
    ws = ws[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         x_ctr + 0.5 * (ws - 1)))
    return anchors


def _scale_enum(anchor, scales):
    w, x_ctr = _whctrs(anchor)
    ws = w * scales
    anchors = _mkanchors(ws, x_ctr)
    return anchors


def generate_anchors(base_size=1, scales=2**np.arange(2, 6)):
    base_anchor = np.array([1, base_size]) - 1
    anchors = _scale_enum(base_anchor, scales)
    return anchors


############################################################
# Evaluations
def _compute_tiou(target_seg, src_segs):
    tt1 = np.maximum(target_seg[0], src_segs[:, 0])
    tt2 = np.minimum(target_seg[1], src_segs[:, 1])

    seg_itsc = (tt2 - tt1).clip(min=0)
    seg_union = (src_segs[:, 1] - src_segs[:, 0]) + \
        (target_seg[1] - target_seg[0]) - seg_itsc

    tiou = seg_itsc.astype(np.float32) / seg_union

    return tiou


# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def compute_metrics(gt, pred, tiou_thresh):
    # Global map for ground truth: each gt could be only assigned to one pred.
    gt_assigned = -1 * np.ones((len(tiou_thresh), len(gt)))

    desc_idx = pred['score'].values.argsort()[::-1]
    pred = pred.loc[desc_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors
    tp = np.zeros((len(tiou_thresh), len((pred))))
    fp = np.zeros((len(tiou_thresh), len((pred))))

    gt_by_videoid = gt.groupby('video-id')

    for idx, cur_pred in pred.iterrows():
        try:
            cur_gt = gt_by_videoid.get_group(cur_pred['video-id'])
        except KeyError:
            fp[:, idx] = 1
            continue

        cur_gt = cur_gt.reset_index()
        cur_tiou = _compute_tiou(cur_pred[['t-start', 't-end']].values,
                                 cur_gt[['t-start', 't-end']].values)
        # Evaluation begins from highest tiou.
        desc_idx = cur_tiou.argsort()[::-1]
        for tidx, cur_thresh in enumerate(tiou_thresh):
            for di in desc_idx:
                if cur_tiou[di] < cur_thresh:
                    fp[tidx, idx] = 1
                    break

                if gt_assigned[tidx, cur_gt.loc[di, 'index']] >= 0:
                    continue

                tp[tidx, idx] = 1
                gt_assigned[tidx, cur_gt.loc[di, 'index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    # Average precision.
    ap = np.zeros(len(tiou_thresh))
    prec = np.zeros_like(tp)
    rec = np.zeros_like(tp)

    # Iterate through different temporal IoU threshold.
    for tidx in range(len(tiou_thresh)):
        cur_tp = np.cumsum(tp[tidx, :]).astype(np.float32)
        cur_fp = np.cumsum(fp[tidx, :]).astype(np.float32)

        cur_prec = cur_tp / (cur_tp + cur_fp)
        cur_rec = 1. * cur_tp / len(gt)

        ap[tidx] = _ap_from_pr(cur_prec, cur_rec)
        prec[tidx] = cur_prec
        rec[tidx] = cur_rec

    return ap, (prec, rec)


############################################################
# Common Operators
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def sigmoid(logits):
    return 1 / (1 + np.exp(-logits))


############################################################
# Video misc
def may_change_infix(s, sep, fld, val):
    base, ext = osp.splitext(s)
    pattern = r'{}{}[0-9\.]*'.format(sep, fld)
    target = '{}{}{}'.format(sep, fld, val)

    if bool(re.search(pattern, base)):
        base = re.sub(pattern, target, base)
    else:
        lst = base.split(sep)
        lst.insert(-1, '{}{}'.format(fld, val))
        base = sep.join(lst)

    return base + ext


def fetch_videoids(phase_key):
    with open(cfg.DSPEC.VID_LST.format(stage=cfg[phase_key].STAGE)) as f:
        videoids = [vid.strip() for vid in f.readlines()]
    return videoids


def load_gt_mat(stage):
    gt_lst = np.load({
        'TH14': cfg.DSPEC.TH14_GT,
        'AN': cfg.DSPEC.AN_GT,
    }[cfg.DATASET].format(stage=stage))

    dtype = [('videoid', str),
             ('start', np.float32),
             ('end', np.float32),
             ('label', int)]

    gt_mat = np.empty((0, 4), dtype=dtype)
    for c in range(len(gt_lst)):
        # df - vid, st, ed.
        gt_df = gt_lst[c]
        gt_df['cls'] = pd.Series(c * np.ones((gt_df.shape[0]), dtype=int),
                                 index=gt_df.index)
        gt_mat = np.append(gt_mat, gt_df.values, axis=0)

    return gt_mat


def load_meta(stage):
    meta = np.load({
        'TH14': cfg.DSPEC.TH14_META,
        'AN': cfg.DSPEC.AN_META,
    }[cfg.DATASET].format(stage=stage))

    return meta


def compute_frame_prior():
    '''Compute prior of frames on validation set. Note that the frames are not
    discrete index but the continuous number before flooring.'''
    meta = np.load({
        'TH14': cfg.DSPEC.TH14_META.format(stage='val'),
        'AN': cfg.DSPEC.AN_META.format(stage='train'),
    }[cfg.DATASET])

    gt_prior_lst = np.load({
        'TH14': cfg.DSPEC.TH14_GT.format(stage='val'),
        'AN': cfg.DSPEC.AN_GT.format(stage='train'),
    }[cfg.DATASET])

    def _row2frame(row):
        key = {
            'TH14': 'video_validation_{}',
            'AN': '{}'
        }[cfg.DATASET].format(row['video-id'])
        action_drt = (row['t-end'] - row['t-start']) * meta[key]['duration']
        fps = meta[key]['fps']

        return action_drt * fps

    frame_prior = np.zeros((cfg.NUM_CLASSES, 2))
    for c in range(cfg.NUM_CLASSES):
        cur_prior = gt_prior_lst[c]
        frames = cur_prior.apply(_row2frame, axis=1).values

        frame_prior[c] = st.norm.fit(frames)

    return frame_prior


def _meta2info(meta, videoid, stage):
    # The gt time is scaled in range [0, 1].
    if cfg.DATASET == 'TH14':
        videoid = int(videoid)
        stage_str = {
            'val': 'validation',
            'test': 'test',
        }[stage]
        key = 'video_{stage}_{num:07d}'.format(stage=stage_str, num=videoid)
    else:
        videoid = int(videoid)
        key = '{num:07d}'.format(num=videoid)

    fps = float(meta[key]['fps'])
    drt = float(meta[key]['duration'])

    return fps * drt


def time2snippet(time, meta, videoid, stage):
    scaled_fps = _meta2info(meta, videoid, stage)
    return np.floor(time.astype(np.float32) * scaled_fps / cfg.DSPEC.WINSTEP)


def snippet2ratio(snippet, meta, videoid, stage):
    scaled_fps = _meta2info(meta, videoid, stage)
    return ((snippet * cfg.DSPEC.WINSTEP + 1) / scaled_fps).clip(0, 1)


############################################################
# Common OS Operations
def may_create(dirname):
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        print 'Create directory at {}'.format(dirname)

    return dirname


def may_create_or_refresh(dirname):
    if not osp.isdir(dirname):
        os.makedirs(dirname)
        print 'Create directory at {}'.format(dirname)
    else:
        for filename in os.listdir(dirname):
            filepath = osp.join(dirname, filename)
            try:
                if osp.isfile(filepath):
                    os.unlink(filepath)
            except Exception as e:
                print e
        print 'Refresh directory at {}'.format(dirname)

    return dirname


def str2bool(s):
    return s.lower() in ['yes', 'true', 't', '1']
