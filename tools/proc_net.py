# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

"""
Script entry for training & testing on TH14/AN dataset.
Usage:
    1. $ mkdir exp/<DATASET>/<EXPNAME>
    2. Setup configuration in `config.yml`
    3. Setup proto template in `solver.tpl` and `train.tpl`
    4. $ python tools/train_net.py --dataset <DATASET> --expname <EXPNAME>
"""

from _init import workenv, setup

import matplotlib
matplotlib.use('Agg')

import caffe
import logging
import argparse
import getpass
import google.protobuf.text_format #noqa

import numpy as np
import pandas as pd
import pickle as pkl
import os.path as osp
import google.protobuf as pb2

from caffe.proto import caffe_pb2

from config import cfg, cfg_from_file
from utils.ops import prll_lock


def train(logger=None):
    '''Training entry for one pair of prototxt setting.'''
    if logger is not None:
        username = getpass.getuser()
        handler = logging.FileHandler('/tmp/{}.train{}.log'.format(username,
                                                                   cfg.INFIX))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Load solver and model.
    solver = caffe.SGDSolver(cfg.SL_PATH)
    # Load solver param
    solver_param = caffe_pb2.SolverParameter()
    with open(cfg.SL_PATH, 'rt') as f:
        pb2.text_format.Merge(f.read(), solver_param)

    while solver.iter < cfg.MAX_ITER:
        solver.step(1)

    trained_net = solver.net
    if cfg.SAVE_MODEL:
        trained_net.save(cfg.SNAPSHOT_PATH)
        print 'Save model at {}'.format(cfg.SNAPSHOT_PATH)

    return trained_net


def test(logger=None):
    '''Testing entry for one pair of prototxt setting.'''
    if logger is not None:
        username = getpass.getuser()
        handler = logging.FileHandler('/tmp/{}.test{}.log'.format(username,
                                                                  cfg.INFIX))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Load pretrained model
    pretrained_net = caffe.Net(cfg.TE_PATH, cfg.SNAPSHOT_PATH, caffe.TEST)
    pretrained_net.forward()

    return pretrained_net


def fetch_rslt(phase_key, net, bbox_pred_name='bbox_pred_rslt'):
    # N x ['video-id', 'cls', 'st', 'ed', 'score']
    pred_rslt = net.blobs[bbox_pred_name].data

    # Note that for now `videoid` is integer, but for SSN code they are using
    #   abosulte path to the video data file.
    rslt = pd.DataFrame(
        pred_rslt,
        columns=['video-id', 'cls', 't-start', 't-end', 'score']
    )
    try:
        # if cfg.DATASET == 'TH14':
        rslt.loc[:, 'video-id'] = rslt.apply(
            lambda row: '{:07d}'.format(int(row['video-id'])), axis=1)
        # else:
            # raise NotImplementedError
    except ValueError:
        # Expected a empty frame when net predicts no bbox.
        pass

    return rslt


def dump_rslt(phase_key, pred_rslts):
    merged_rslt = pd.concat(pred_rslts, axis=0, ignore_index=True)

    rslt_by_cls = [
        merged_rslt[merged_rslt['cls'] == c]. \
        reset_index(drop=True) for c in range(cfg.NUM_CLASSES)
    ]

    pkl.dump(rslt_by_cls,
             open(cfg[phase_key].RSLT_PATH, 'wb+'),
             pkl.HIGHEST_PROTOCOL)
    print 'Dump results at {}'.format(cfg[phase_key].RSLT_PATH)

    return rslt_by_cls


if __name__ =="__main__":
    with workenv():
        old_settings = np.seterr(all='raise', under='ignore')

        parser = argparse.ArgumentParser()

        parser.add_argument('--phase', type=str, required=True,
                            choices=['train', 'test'])
        parser.add_argument('--dataset', type=str, required=True,
                            choices=['TH14', 'AN'])
        parser.add_argument('--expname', type=str, required=True)
        parser.add_argument('--rsltname', type=str, default='rslt')
        # This argument would only be used when testing.
        parser.add_argument('--pretrained', type=str,
                            default='default.caffemodel')

        args = parser.parse_args()

        config_path = osp.join(cfg.EXP_DIR, args.dataset,
                               args.expname, 'config.yml')

        # Init
        cfg_from_file(config_path)
        args.phase = args.phase.upper()
        setup(args.phase, args.dataset, args.expname, args.rsltname)

        if args.phase == 'TEST':
            # Check whether specified <pretrained> when testing:
            # a work-around for conditional required argument.
            assert args.pretrained != 'default.caffemodel', \
                'Speicify pretrained model when testing.'
            cfg.SNAPSHOT_PATH = osp.join(cfg.LOCAL_SNAPSHOT_PATH,
                                         args.pretrained)

        caffe.set_device(cfg.GPU_ID)
        caffe.set_mode_gpu()
        caffe.init_glog(osp.join(cfg.LOG_PATH, '{}.{}.'.format(args.phase,
                                                               cfg.INFIX)))

        prll_lock.acquire()

        runner = eval(args.phase.lower())
        net = runner()

        pred_rslts = [fetch_rslt(args.phase, net)]
        dump_rslt(args.phase, pred_rslts)
