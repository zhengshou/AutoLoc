# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import caffe

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats import norm

from config import cfg
from utils.viz import plot_bbox
from utils.postproc import proc_heatmap_att, nms_1d
from utils.ops import may_create_or_refresh, snippet2ratio, \
    load_gt_mat, load_meta, compute_frame_prior


'''
Current setting:
    - singleimage
    - multiclass
    - maxanchor
    - clipkeepmg
'''
class OuterInnerContrastiveLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) == 6,    'requires 6 bottom blobs'
        assert len(top) == 3,       'requires 2 top blobs'

        #self._phase_key = str(self.phase) # either 'TRAIN' or 'TEST'
        if self.phase == 0: self._phase_key = 'TRAIN'
        if self.phase == 1: self._phase_key =  'TEST'

        self._stage = cfg[self._phase_key].STAGE
        self._scale = cfg[self._phase_key].FEAT_SCALE
        self._plot_itvl = cfg.VIZ.PLOT_ITVL

        self._fg_thresh = cfg[self._phase_key].FG_THRESH
        self._oic_loss_thresh = cfg[self._phase_key].OIC_LOSS_THRESH

        self._num_acr = bottom[5].data.shape[1] / 2
        # For single optimization.
        # self._max_anchor = -1*np.ones(self._gt_label_classes.shape[1])

        self._gt_mat = load_gt_mat(self._stage)
        self._meta = load_meta(self._stage)
        self._frame_prior = compute_frame_prior()

        # Keep track of seen videoids
        self._seen_videoids = defaultdict(bool)

        self._num_iter = 0

        # (k, a, x) for optimization, shared with back-prop.
        self._ax2opt = defaultdict(tuple)

        top[0].reshape(1)
        top[1].reshape(1, 1, 1, 10)
        top[2].reshape(1, 5)

    def region_sum(self, int_heatmap, x1, x2):
        '''Calculate region sum for [x1,x2] with start&end included.'''
        left = int_heatmap[np.minimum(int(x1) - 1, int_heatmap.shape[0]-1)] if x1 > 0 else 0
        right = int_heatmap[np.minimum(int(x2), int_heatmap.shape[0]-1)] if x2 > 0 else 0

        return right - left

    def point_sum(self, int_heatmap, x):
        '''Calculate region sum for [x1,y1,x2,y2] with start&end included.'''
        # Mimic line_sum. The goal is to find the true value of f(x)
        # when x is out of range, should return 0. the input value of x should
        # not be clipped which will cause to fall into interval of non-zero values.
        x_max = int_heatmap.shape[0] - 1
        if x>= 0 and x<=x_max:
            return int_heatmap[int(x)]
        else:
            return 0

    def region_area(self, x1, x2):
        return x2 - x1 + 1

                # Update oic_loss

    def oic_from_pred(self, inner, outer, intmap, a, x):
        x1, x2 = inner[0, a::self._num_acr, 0, x]
        i_sum = self.region_sum(intmap, x1, x2)
        i_area = self.region_area(x1, x2)

        X1, X2 = outer[0, a::self._num_acr, 0, x]
        o_sum = self.region_sum(intmap, X1, X2)
        o_area = self.region_area(X1, X2)

        if o_area != i_area:
            Lo = (o_sum - i_sum) / (o_area - i_area)
            Li = i_sum / i_area
            # Fit loss ranged in [0, 1].
            oic = (Lo - Li + 1) / 2
        else:
            oic = 1

        return oic, (x1, x2), (X1, X2)

    def forward(self, bottom, top):
        self._gt_label_classes = bottom[0].data.astype(np.int)
        self._att = bottom[2].data

        cur_videoid = int(bottom[3].data[0, 0])
        cur_fig_dir = osp.join(cfg.VIZ_PATH, '{:07d}'.format(cur_videoid))

        if not self._seen_videoids[cur_videoid]:
            may_create_or_refresh(cur_fig_dir)
            self._seen_videoids[cur_videoid] = True

        self._count = 0
        loss = 0

        # Number of classes
        K = bottom[1].data.shape[1]
        # Number of anchors
        A = self._num_acr
        # Number of snippets
        T = bottom[1].data.shape[3]

        # Anchor score collection for this pass: A x T x [st, ed, score]
        self._acr_scores = np.zeros((K, A, T, 3), dtype=np.float32)

        # Unfloored frame float index to caculate refined scores.
        inner_frames = (bottom[4].data * self._scale * cfg.DSPEC.WINSTEP). \
            reshape(-1, A, T)
        # Snippet index
        inner = np.floor(bottom[4].data * self._scale)
        outer = np.floor(bottom[5].data * self._scale)

        heat_per_class = bottom[1].data[0, :, 0].mean(axis=1)
        ks = {
            True: self._gt_label_classes[0, :],
            False: heat_per_class.argsort()[::-1][:cfg[self._phase_key].OIC_CLS_TOPK],
        }[cfg[self._phase_key].OIC_CLS_TOPK == -1]

        bbox_pred_rslt = np.empty((0, 5), dtype=np.float32)
        for k in ks:
            heatmap = bottom[1].data[0, k, 0]
            heatmap_ = proc_heatmap_att(heatmap,
                                        self._att[0, 0, 0],
                                        self._att[0, 1, 0])

            cur_intmap = heatmap_.cumsum(axis=0)
            if cfg.DEBUG:
                ''' Debug starts '''
                print "[DEBUG]{:-^40}".format(self.__class__.__name__)
                print "heatmap:"
                print heatmap
                print "processed heatmap:"
                print heatmap
                print "heatmap integration:"
                print cur_intmap
                ''' Debug ends '''

            xs = range(T) if cfg.LSPEC.OIC_POS_OPT == 'mlt' else [heatmap_.argmax()]

            top[1].data[...] = 0

            for a in range(A):
                for x in xs:
                    cur_loss, (x1, x2), _ = self.oic_from_pred(inner,
                                                               outer,
                                                               cur_intmap,
                                                               a, x)

                    # All scores are in [0, 1], the higher the better.
                    cur_score = 1 - cur_loss
                    top[1].data[0, k, a, x] = cur_score

                    self._acr_scores[k, a, x] = np.asarray([x1, x2, cur_score])

            bbox2nms = []
            bbox2ax = {}
            for x in xs:
                cur_prior = self._frame_prior[k]
                cur_scores = self._acr_scores[k, :, x, 2]
                if heatmap_[x] > self._fg_thresh:
                    if cfg.LSPEC.OIC_ANC_OPT == 'org':
                        a = np.argmax(cur_scores)
                    else:
                        len_frame = inner_frames[1, :, x] - inner_frames[0, :, x]
                        ref_scores = cur_scores * norm.pdf(len_frame, *cur_prior)
                        a = np.argmax(ref_scores)

                    if self._acr_scores[k, a, x, 2] > 1 - self._oic_loss_thresh:
                        bbox2nms.append(self._acr_scores[k, a, x])
                        bbox2ax[tuple(bbox2nms[-1])] = a, x

            nms_bbox = nms_1d(np.array(bbox2nms),
                              cfg[self._phase_key].NMS_BBOX_THRESH)
            self._ax2opt[k] = [bbox2ax[tuple(nb)] for nb in nms_bbox]

            bboxes_loss = np.empty((0), np.float32)
            bboxes_inner = np.empty((0, 2), int)
            bboxes_outer = np.empty((0, 2), int)

            for (a, x) in self._ax2opt[k]:
                bbox_score = top[1].data[0, k, a, x]
                bbox_inner = inner[0, a::self._num_acr, 0, x]
                bbox_outer = outer[0, a::self._num_acr, 0, x]

                inner_ratio = snippet2ratio(bbox_inner,
                                            self._meta,
                                            cur_videoid,
                                            self._stage)
                # print inner_ratio, k

                # [vid, label, x1, x2, score]...
                cur_rslt = np.array([cur_videoid, k] + \
                                    inner_ratio.tolist() + \
                                    [bbox_score]).reshape(1, -1)

                bbox_pred_rslt = np.append(bbox_pred_rslt,
                                           cur_rslt, axis=0)

                if cfg.DEBUG:
                    ''' Debug starts '''
                    print "[DEBUG]{:-^40}".format(self.__class__.__name__)
                    print "F anchor index:"
                    print a
                    print "F oic_score:"
                    print bbox_score
                    print "F bbox_inner x1 x2:"
                    print inner[0, a::self._num_acr, 0, x]
                    # print inner[0, 0::self._num_acr, 0, 24]
                    print "F bbox_outer x1 x2:"
                    print inner[0, a::self._num_acr, 0, x]
                    # print outer[0, 0::self._num_acr, 0, 24]
                    ''' Debug ends '''

                # Update total loss for this video
                cur_loss = 1 - self._acr_scores[k, a, x, 2]
                loss +=  cur_loss
                self._count += 1

                bboxes_loss = np.append(bboxes_loss, np.array([cur_loss]), axis=0)
                bboxes_inner = np.append(bboxes_inner, bbox_inner.reshape(1, -1), axis=0)
                bboxes_outer = np.append(bboxes_outer, bbox_outer.reshape(1, -1), axis=0)
                # print bbox_inner

            if self._num_iter % self._plot_itvl == 0 or \
                    self._num_iter == cfg.MAX_ITER - 1:
                # Visualize prediction result for each gt classes.
                plot_bbox(heatmap_, bboxes_inner, bboxes_outer,
                          self._gt_mat, self._meta,
                          cur_videoid, k, self._stage, bboxes_loss)

                plt.savefig(osp.join(
                    cur_fig_dir,
                    'vid{:07d}_label{:02d}_iter{:04d}.jpg'. \
                    format(cur_videoid, k, self._num_iter))
                )

        top[0].data[0] = loss / self._count if self._count > 0 else 1
        # We don't need to change top[1]
        top[2].reshape(*bbox_pred_rslt.shape)
        top[2].data[...] = bbox_pred_rslt

        self._num_iter += 1

    def backward(self, top, propagate_down, bottom):
        self._gt_label_classes = bottom[0].data.astype(np.int)

        loss_weight = top[0].diff[0] # see train.prototxt
        # We would only update inner bound.
        inner_diff = bottom[4].diff
        inner_diff[:] = np.zeros(inner_diff.shape, dtype=np.float32)
        # update outer also
        outer_diff = bottom[5].diff
        outer_diff[:] = np.zeros(outer_diff.shape, dtype=np.float32)

        inner = np.floor(bottom[4].data * self._scale)
        outer = np.floor(bottom[5].data * self._scale)

        for k in self._gt_label_classes[0, :]:
            heatmap = bottom[1].data[0, k, 0]
            heatmap_ = proc_heatmap_att(heatmap,
                                        self._att[0, 0, 0],
                                        self._att[0, 1, 0])

            cur_intmap = heatmap_.cumsum(axis=0)

            for (a, x) in self._ax2opt[k]:
                x1, x2 = inner[0, a::self._num_acr, 0, x]
                i_sum = self.region_sum(cur_intmap, x1, x2)
                i_area = self.region_area(x1, x2)

                X1, X2 = outer[0, a::self._num_acr, 0, x]
                o_sum = self.region_sum(cur_intmap, X1, X2)
                o_area = self.region_area(X1, X2)

                if o_area != i_area:
                    Lo = (o_sum - i_sum) / (o_area - i_area)
                    Li = i_sum / i_area

                    dx1 = (self.point_sum(heatmap_, x1) - Lo) / (o_area - i_area) - \
                        (-self.point_sum(heatmap_, x1) + Li) / i_area
                    dx2 = (-self.point_sum(heatmap_, x2) + Lo) / (o_area - i_area) - \
                        (self.point_sum(heatmap_, x2) - Li) / i_area

                    dX1 = (-self.point_sum(heatmap_, X1) + Lo) / (o_area - i_area)
                    dX2 = (self.point_sum(heatmap_, X2) - Lo) / (o_area - i_area)

                    inner_diff[0, a::self._num_acr, 0, x] += 0.5 * \
                        np.array((dx1, dx2)) * self._scale / self._count * loss_weight

                    outer_diff[0, a::self._num_acr, 0, x] += 0.5 * \
                        np.array((dX1, dX2)) * self._scale / self._count * loss_weight

                    if cfg.DEBUG:
                        ''' Debug starts '''
                        print "[DEBUG]{:-^40}".format(self.__class__.__name__)
                        print "B inner_diff:"
                        # print inner_diff[0, a::self._num_acr, 0, x]
                        print inner_diff[0, 0::self._num_acr, 0, 24]
                        print "B max diff:"
                        print 'inner: {} at {}'.format(inner_diff.max(),
                                                       inner_diff.argmax())
                        print "B min diff:"
                        print 'inner: {} at {}'.format(inner_diff.min(),
                                                       inner_diff.argmin())
                        ''' Debug ends '''

    def reshape(self, bottom, top):
        # heatmap.shape: [1, K, 1, T]
        s = bottom[1].data.shape
        top[1].reshape(s[0], s[1], self._num_acr, s[3])
