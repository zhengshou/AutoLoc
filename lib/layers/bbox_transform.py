# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import caffe
import json

import numpy as np

from config import cfg
from utils.ops import generate_anchors


class BboxTransformLayer(caffe.Layer):
    '''Bbox transform layer to convert delta (dx,dw) to (x,w).'''
    def setup(self, bottom, top):
        assert len(bottom) == 1,    'requires 1 bottom blob'
        assert len(top) == 1,       'requires 1 top blob'

        #phase_key = str(self.phase) # either 'TRAIN' or 'TEST'
        if self.phase == 0: phase_key = 'TRAIN'
        if self.phase == 1: phase_key =  'TEST'

        self._anchors = generate_anchors(base_size=1,
                                         scales=np.array(
                                             cfg[phase_key].ANCHOR_SCALES)
                                         )
        self._feat_stride = cfg[phase_key].FEAT_STRIDE

        self._anchor_w = self._anchors[:, 1] - self._anchors[:, 0] + 1
        self._num_acr = self._anchors.shape[0]

        # Rois blob: holds R regions of interest, each is a 3-tuple
        # (n, x1, x2) specifying an image batch index n and a
        # rectangle (x1, x2)
        top[0].reshape(1, 3)

        # Scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1)

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        delta = bottom[0].data
        width = delta.shape[-1]
        shift_x = np.arange(0, width) * self._feat_stride

        x1 = np.zeros([self._num_acr, width], dtype=np.float32)
        x2 = np.zeros([self._num_acr, width], dtype=np.float32)

        for a in np.arange(self._num_acr):
            cx = delta[:, self._num_acr*0+a, ...] * self._anchor_w[a] + shift_x
            w = np.exp(delta[:, self._num_acr*1+a, ...]) * self._anchor_w[a]

            x1[a, ...] = cx - w / 2
            x2[a, ...] = cx + w / 2

            if cfg.DEBUG and a == 0:
                ''' Debug starts '''
                print "[DEBUG]{:-^40}".format(self.__class__.__name__)
                print "F pred anchor 0 -- cx, w:"
                print cx[0,0,24], w[0,0,24]
                ''' Debug ends '''

        if cfg.DEBUG:
            ''' Debug starts '''
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            print 'num_anchors: {}'.format(self._num_acr)
            print "F pred anchor 0 -- delta x w:"
            print delta[0,0::self._num_acr,0,24]
            print "F anchor 0 -- x1 x2:"
            print [x1[0,24],x2[0,24]]
            ''' Debug ends '''

        top[0].data[0,...] = np.vstack((np.expand_dims(x1, axis=1),
                                        np.expand_dims(x2, axis=1)))

    def backward(self, top, propagate_down, bottom):
        top_diff = top[0].diff
        delta = bottom[0].data

        delta_diff = bottom[0].diff

        for a in np.arange(self._num_acr):
            # \partial{L}{dx}
            delta_diff[:,self._num_acr*0+a,...] = (
                top_diff[:,self._num_acr*0+a,...] +
                top_diff[:,self._num_acr*1+a,...]) * self._anchor_w[a]
            # \partial{L}{dw}
            delta_diff[:,self._num_acr*1+a,...] = (
                top_diff[:,self._num_acr*1+a,...] -
                top_diff[:,self._num_acr*0+a,...]) * 0.5 * \
                np.exp(delta[:,self._num_acr*1+a,...]) * self._anchor_w[a]

        if cfg.DEBUG:
            ''' Debug starts '''
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            print "B pred anchor 0 -- delta x w:"
            print delta[0,0::self._num_acr,0,24]
            print "B anchor 0 -- top \partial x1 x2:"
            print top_diff[0,0::self._num_acr,0,24]
            print "B anchor 0 -- \partial x w:"
            print delta_diff[0,0::self._num_acr,0,24]
            ''' Debug ends '''

    def reshape(self, bottom, top):
        top[0].reshape(*(bottom[0].data.shape))


class BboxInflateRatioMinLayer(caffe.Layer):
    '''
    Bbox inflate layer to inflate (x1,x2) to (x1-a,x2+a)
    '''
    def setup(self, bottom, top):
        assert len(bottom) == 1,    'requires 1 bottom blob'
        assert len(top) == 1,       'requires 1 top blob'

        #phase_key = str(self.phase) # either 'TRAIN' or 'TEST'
        if self.phase == 0: phase_key = 'TRAIN'
        if self.phase == 1: phase_key =  'TEST'

        self._outer_inflate_ratio = cfg[phase_key].OUTER_INFLATE_RATIO
        self._outer_min = cfg[phase_key].OUTER_MIN

        self._num_acr = bottom[0].data.shape[1] / 2

        top[0].reshape(*(bottom[0].data.shape))

    def forward(self, bottom, top):
        inflate_w = (bottom[0].data[:, self._num_acr*1:self._num_acr*2, ...] -
                     bottom[0].data[:, self._num_acr*0:self._num_acr*1, ...]
                     ) * self._outer_inflate_ratio

        inflate_w[np.where(inflate_w < self._outer_min)] = self._outer_min

        top[0].data[:, self._num_acr*0:self._num_acr*1, ...] = \
            bottom[0].data[:, self._num_acr*0:self._num_acr*1, ...] - \
            inflate_w
        top[0].data[:, self._num_acr*1:self._num_acr*2, ...] = \
            bottom[0].data[:, self._num_acr*1:self._num_acr*2, ...] + \
            inflate_w

        if cfg.DEBUG:
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            ''' Debug starts '''
            print "F Inflate anchor 0 -- x1 x2:"
            print top[0].data[0,0::self._num_acr,0,24]
            ''' Debug ends '''

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff

    def reshape(self, bottom, top):
        top[0].reshape(*(bottom[0].data.shape))


class BboxClipKeepGradMgLayer(caffe.Layer):
    '''
    Bbox clip layer to clip (x1,x2) to be within (0,v_width - 1)
    '''
    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires 2 bottom blobs: bbox and v_info'
        assert len(top) == 1, 'requires 1 top blobs: bbox'

        #phase_key = str(self.phase) # either 'TRAIN' or 'TEST'
        if self.phase == 0: phase_key = 'TRAIN'
        if self.phase == 1: phase_key =  'TEST'

        assert hasattr(self, 'param_str'), 'requires param_str'
        params = json.loads(self.param_str)

        # Corner case when bbox falls outside of video boundary.
        self._margin = params['margin'] / cfg[phase_key].FEAT_SCALE

        top[0].reshape(*(bottom[0].data.shape))
        self._num_acr = bottom[0].data.shape[1] / 2

    def clip(self, x, v_width):
        return np.maximum(np.minimum(x, v_width - 1 + self._margin),
                          -self._margin)

    def forward(self, bottom, top):
        v_width = bottom[1].data[0, 0]

        top[0].data[:,self._num_acr*0:self._num_acr*1,...] = \
            self.clip(bottom[0].data[:,self._num_acr*0:self._num_acr*1,...],
                      v_width)
        top[0].data[:,self._num_acr*1:self._num_acr*2,...] = \
            self.clip(bottom[0].data[:,self._num_acr*1:self._num_acr*2,...],
                      v_width)

        if cfg.DEBUG:
            ''' Debug starts '''
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            print "F before clip anchor 0 -- x1 x2:"
            print bottom[0].data[0,0::self._num_acr,0,24]
            print "F after clip anchor 0 -- x1 x2:"
            print top[0].data[0,0::self._num_acr,0,24]
            ''' Debug ends '''

    def backward(self, top, propagate_down, bottom):

        if cfg.DEBUG:
            ''' Debug starts '''
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            print "B clip anchor 0 -- x1 x2 before process:"
            print bottom[0].diff[0,0::self._num_acr,0,24]
            ''' Debug ends '''

        bottom[0].diff[:,self._num_acr*0:self._num_acr*1,...] = \
            top[0].diff[:,self._num_acr*0:self._num_acr*1,...]
        bottom[0].diff[:,self._num_acr*1:self._num_acr*2,...] = \
            top[0].diff[:,self._num_acr*1:self._num_acr*2,...]

        if cfg.DEBUG:
            ''' Debug starts '''
            print "[DEBUG]{:-^40}".format(self.__class__.__name__)
            print "B clip anchor 0 -- x1 x2:"
            print bottom[0].diff[0,0::self._num_acr,0,24]
            ''' Debug ends '''

    def reshape(self, bottom, top):
        top[0].reshape(*(bottom[0].data.shape))
