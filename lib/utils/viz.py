# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from config import cfg
from ops import time2snippet


_margin = None


def _select_gt_by_context(gt_mat, videoid, label):
    '''Select ground truth [[vid, lb, st, ed]] by context (vid, lb).'''
    videoids = gt_mat[:, 0]
    mask = (videoids == '{:07d}'.format(int(videoid))).reshape(-1)

    gt_mat_ = gt_mat[mask]
    labels = gt_mat_[:, 3]

    mask = (labels == label).reshape(-1)

    return gt_mat_[mask]


def _barplot_fix_width(heatmap, width=0.5):
    n = len(heatmap)
    plt.bar(range(n), heatmap, width=width)

    # add loads of ticks
    plt.xticks(range(0, n, 5), range(0, n, 5))

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    # inch margin
    m = 0.3
    s = 1. * maxsize / plt.gcf().dpi * n + 2. * m

    max_constrain = 2**16
    s /= np.ceil(100 * s / max_constrain)

    global _margin
    if _margin is None:
        _margin = 1. * m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=_margin, right=1.-_margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])


def plot_bbox(heatmap, bbox_inner, bbox_outer,
              gt_mat, meta,
              videoid, label, stage, loss=None):
    m = np.floor(heatmap.min())
    n = np.ceil(heatmap.max())
    # n = np.minimum(heatmap.max() * 1.2, np.ceil(heatmap.max()))

    def _plot_bound(bound, c, height_aspect, loss=None):
        if bound is None:
            return
        # Plot bounds recursively
        if len(bound.shape) == 1:
            delta = (n - m) * (1 - height_aspect) / 2.
            m_ = m + delta
            n_ = n - delta

            plt.gca().add_patch(
                plt.Rectangle((bound[0], m_),
                              bound[1] - bound[0],
                              n_ - m_, fill=False,
                              edgecolor=c, linewidth=1.5)
            )
            if loss is not None and cfg.VIZ.FIX_WIDTH:
                w, h = 0.8, (n_ - m_) / 5.
                plt.gca().add_patch(
                    plt.Rectangle((bound[0], n_-h),
                                w,
                                h, fill=True,
                                facecolor=c, linewidth=1.5)
                )
                plt.gca().annotate('{:1.4f}'.format(loss),
                                   (bound[0]+w/2., n_-h/2.),
                                   color='w', weight='bold', rotation=90,
                                   fontsize=6, ha='center', va='center')
        else:
            if loss is None:
                for idx in range(bound.shape[0]):
                    height_aspect_ = (height_aspect - 1) * np.random.random() + 1
                    _plot_bound(bound[idx], c, height_aspect_, loss=loss)
            else:
                assert bound.shape[0] == loss.shape[0], 'Inconsistent shape'
                for idx in range(bound.shape[0]):
                    height_aspect_ = (height_aspect - 1) * np.random.random() + 1
                    _plot_bound(bound[idx], c, height_aspect_, loss=loss[idx])

    def _plot_gtbg(bound, c):
        '''Plot ground truth as background rectangle.'''
        for i in range(bound.shape[0]):
            plt.gca().add_patch(
                plt.Rectangle((bound[i, 0], m),
                                bound[i, 1] - bound[i, 0],
                                n - m, fill=True,
                                facecolor=c, alpha=.5, linewidth=1)
            )

    gt = _select_gt_by_context(gt_mat, videoid, label)
    snippet_gt = time2snippet(gt[:, 1:3], meta, videoid, stage)

    # Plot heatmap according to experiment setting.
    plt.cla()
    if cfg.VIZ.FIX_WIDTH:
        _barplot_fix_width(heatmap)
    else:
        plt.bar(range(len(heatmap)), heatmap)

    plt.gca().set_ylim([m - 0.08 * (n - m),
                        n + 0.08 * (n - m)])
    plt.gca().set_xlim([-3, len(heatmap) + 2])
    plt.grid()

    # Plot {inner, outer}_pred and ground truth.
    _plot_gtbg(snippet_gt, 'tan')
    _plot_bound(bbox_inner, 'g', 1.05, loss)
    _plot_bound(bbox_outer, 'r', 1.1)


def plot_pr_curve(prec_by_class, rec_by_class, tiou_thresh, prc_path):
    fig, _ = plt.subplots(ncols=cfg.VIZ.PLOT_PR_NCOLS,
                          figsize=(cfg.VIZ.PLOT_PR_NCOLS*4,
                                   cfg.NUM_CLASSES/cfg.VIZ.PLOT_PR_NCOLS*3))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    for clazz in range(cfg.NUM_CLASSES):
        plt.subplot(cfg.NUM_CLASSES / cfg.VIZ.PLOT_PR_NCOLS,
                    cfg.VIZ.PLOT_PR_NCOLS, clazz + 1)

        for tidx, cur_thresh in enumerate(tiou_thresh):
            plt.plot(rec_by_class[clazz][tidx],
                     prec_by_class[clazz][tidx],
                     label='t-IoU: {}'.format(cur_thresh))

        plt.title('class {}'.format(clazz))
        plt.xlabel('rec')
        plt.ylabel('prec')
        plt.grid()
        plt.ylim(-0.1, 1.1)

    fig.suptitle('PR curve for each class under different t-IoU',
                    fontsize=16)
    lgd = fig.legend(loc='right', ncol=1)

    plt.savefig(prc_path, additional_artists=[lgd])
    print 'PR Curve saved at {}'.format(prc_path)
