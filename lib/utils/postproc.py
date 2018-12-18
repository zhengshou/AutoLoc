# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import numpy as np

from config import cfg
from utils.ops import softmax, sigmoid


def proc_heatmap_att(hm, att_temp, att_spac, *args, **kwargs):
    def _relu(logits):
        if cfg.FEAT.MODE == 'mul':
            return np.array([l if l > 0 else 0 for l in logits])
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits])
    def _softmax(logits):
        logits_ = softmax(logits)
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])
    def _sigmoid(logits):
        logits_ = sigmoid(logits)
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])
    def _sigmoid2(logits):
        logits_ = sigmoid(sigmoid(logits))
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])
    def _sigmoid_2(logits):
        logits_ = np.square(sigmoid(logits))
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])
    def _inv(logits):
        logits_ = 1/(1+np.abs(logits))
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])
    def _tanh(logits):
        logits_ = np.tanh(logits)
        if cfg.FEAT.MODE == 'mul':
            return logits_
        else:
            return np.array([1 if l > cfg.FEAT.THRESH else 0 for l in logits_])

    func = eval('_{}'.format(cfg.FEAT.ACTIVATION))
    mask_temp, mask_spac = func(att_temp), func(att_spac)

    hm_ = hm * (mask_temp + mask_spac) * .5

    if cfg.FEAT.NORM:
        if hm_.max() == hm_.min():
            hm_ = np.zeros_like(hm_)
        else:
            hm_ = (hm_ - hm_.min()) /  (hm_.max() - hm_.min())

    return hm_


def nms_1d(bboxes, tiou_thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :return:
    """
    if bboxes.size == 0:
        return np.empty((0, 3))

    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        iou = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        idxs = np.where(iou <= tiou_thresh)[0]
        order = order[idxs + 1]

    return bboxes[keep, :]
