# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

import os
import caffe
import random

import numpy as np

from config import cfg
from utils.ops import prll_lock


class tsvdb():
    '''Feature database in TSV format.'''
    def __init__(self, tsv_filename, col_feature,
                 col_label, col_heatmap, col_att):
        '''
        folder_name: the root folder of the tsv dataset
        tsv_file_name: tsv file name
        '''
        # Use this dict for storing dataset specific config options
        self.config = {}
        self._tsv_filename = tsv_filename
        self._col_feature = col_feature
        self._col_label = col_label
        self._col_heatmap = col_heatmap
        self._col_att = col_att

        assert os.path.exists(self._tsv_filename), \
            'Tsv file does not exist: {}'.format(self._tsv_filename)
        self._tsv_f = open(self._tsv_filename, 'r')

        # load lineidx file for random access
        lineidx_file = os.path.splitext(self._tsv_filename)[0] + '.lineidx'
        with open(lineidx_file, 'r') as f:
            self._lineidx = [int(line.split('\t')[0]) for line in f]

        self._line_shuffle = list(range(len(self._lineidx)))
        random.shuffle(self._line_shuffle)

    def video_data_at(self, i, num_classes, num_feat):
        '''
        Return image i in the image sequence.
        i is the index in the tsv file
        '''
        # find the line no first
        line_no = self._line_shuffle[i]
        # seek to the beginning of that line
        self._tsv_f.seek(self._lineidx[line_no], 0)
        # read the line content
        line = self._tsv_f.readline().rstrip()
        cols = line.split('\t')

        # Only for TH14
        videoid = np.reshape(np.array(cols[0], dtype=int), (1, -1))
        # Labels are all 1-indexed in data files.
        singlelabel = np.reshape(np.fromstring(cols[self._col_label],
                                               dtype=int, sep=';'), (1, -1)) - 1

        video_feature = np.fromstring(cols[self._col_feature],
                                      dtype=np.float32, sep=';')
        video_heatmap = np.fromstring(cols[self._col_heatmap],
                                      dtype=np.float32, sep=';')
        video_att = np.fromstring(cols[self._col_att],
                                      dtype=np.float32, sep=';')

        num_snippets = len(video_feature) / num_feat
        _num_snippets = len(video_heatmap) * .5 /num_classes
        assert num_snippets == _num_snippets, 'Snippet length cannot match.'

        v_info = np.reshape(np.array(num_snippets), (1, -1))

        # (1024 + 1024) x T
        video_feature = np.reshape(video_feature, (1, -1, 1, num_feat))
        video_feature = np.swapaxes(video_feature, 1, 3)

        # K x T
        video_heatmap = np.reshape(video_heatmap, (1, -1, 1, num_classes))
        # Fuse temporal and spatial heatmap.
        temp_heatmap= video_heatmap[:, ::2, ...]
        spac_heatmap= video_heatmap[:, 1::2, ...]
        video_heatmap = .5 * (temp_heatmap + spac_heatmap)

        video_heatmap = np.swapaxes(video_heatmap, 1, 3)

        # 2 x T
        video_att = np.reshape(video_att, (1, -1, 1, 2))
        video_att = np.swapaxes(video_att, 1, 3)

        return video_feature, v_info, singlelabel, video_heatmap, \
            video_att, videoid

    @property
    def _num_videos(self):
        return len(self._line_shuffle)


class TSVVideoDataLayer(caffe.Layer):
    '''
    TSV data layer for video with pre-extracted features and associated heatmaps
    '''
    def setup(self, bottom, top):
        assert len(top) == 6, 'requires 6 top blobs'

        #phase_key = str(self.phase) # either 'TRAIN' or 'TEST'
        if self.phase == 0:
            phase_key = 'TRAIN'
            print 'phase_key: ' + phase_key
        if self.phase == 1:
            phase_key =  'TEST'
            print 'phase_key: ' + phase_key

        self._num_classes = cfg.NUM_CLASSES
        self._num_feat = cfg.NUM_FEAT

        # May modify <data_path> in parallel training, release lock when
        # video data is loaded in the data layer.
        self._tsvdb = tsvdb(cfg[phase_key].DATA_PATH,
                            cfg[phase_key].COL_FEAT,
                            cfg[phase_key].COL_LABEL,
                            cfg[phase_key].COL_HEATMAP,
                            cfg[phase_key].COL_ATT)
        prll_lock.release()


        self._cur = 0

        # Init top layer's shape as placeholder, possible to be overriden
        # when forwarding.
        top[0].reshape(1, self._num_feat, 1, 1)  # feature
        top[1].reshape(1, 1) # v_info: video's length
        top[2].reshape(1, 1) # label
        top[3].reshape(1, self._num_classes, 1, 1) # heatmap
        top[4].reshape(1, 2, 1, 1) # att
        top[5].reshape(1, 1) # videoid

    def _fetch_next_batch(self):
        blobs = self._tsvdb.video_data_at(self._cur,
                                          self._num_classes,
                                          self._num_feat)
        self._cur = (self._cur + 1) % self._tsvdb._num_videos

        return blobs

    def forward(self, bottom, top):
        blobs = self._fetch_next_batch()

        for i in range(6):
            # Reshape net's input blobs
            top[i].reshape(*(blobs[i].shape))
            # Copy data into net's input blobs
            top[i].data[...] = blobs[i].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        '''This layer does not propagate gradients.'''
        pass

    def reshape(self, bottom, top):
        '''Reshaping happens during the call to forward.'''
        pass
