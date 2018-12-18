# -------------------------------------------------------------------------------------
# AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos. ECCV'18.
# Authors: Zheng Shou, Hang Gao, Lei Zhang, Kazuyuki Miyazawa, Shih-Fu Chang.
# -------------------------------------------------------------------------------------

"""
Script entry for training & testing on TH14/AN dataset in parallel mode.
Usage:
    1. $ mkdir exp/<DATASET>/<EXPNAME>
    2. Setup configuration in `config.yml`
    3. Setup proto template in `solver.tpl` and `train.tpl`
    4. $ python tools/train_prll_net.py --dataset <DATASET> \
        --expname <EXPNAME> --num_workers <NUM_WORKERS>
"""

from _init import workenv, setup

import matplotlib
matplotlib.use('Agg')

import caffe
import logging
import argparse

import numpy as np
import os.path as osp

from multiprocessing import Pool, cpu_count

from config import cfg, cfg_from_file
from utils.ops import fetch_videoids, prll_lock, may_change_infix
from proc_net import train, test, fetch_rslt, dump_rslt #noqa


def proc_new_video(phase_key, videoid, logger=None):
    prll_lock.acquire()

    caffe.set_device(cfg.GPU_ID)
    caffe.set_mode_gpu()

    # Change <_data_file_> in memory
    lst = cfg[phase_key].DATA_PATH.split('.')
    lst[-2] = videoid
    cfg[phase_key].DATA_FILE = '.'.join(lst)

    cfg[phase_key].DATA_PATH = osp.join(cfg.DATA_DIR, cfg.DATASET,
                                        cfg[phase_key].DATA_FILE)

    # Would only modify <logger_path> and <snapshot_path>
    cfg.INFIX = '.{}'.format(videoid)

    # Change path to save model in training phase by changing infix.
    # Example: snapshot/rslt_trainnms0.7_by0000981_iter1.caffemodel
    if phase_key == 'TRAIN':
        cfg.SNAPSHOT_PATH = \
            may_change_infix(cfg.SNAPSHOT_PATH, '_', 'by', videoid)

    runner = eval(phase_key.lower())
    net = runner(logger)

    return net


def rslt_wrapper(phase_key, videoid, logger=None):
    '''Wrapper for net object, since multiprocessing.apply_async would
    fail silently if job returns unserializable.'''
    net = proc_new_video(phase_key, videoid, logger)
    rslt = fetch_rslt(phase_key, net)

    return rslt


def proc_prll(phase_key, num_workers):
    pred_rslts = []
    def _append_rslt(rslt):
        pred_rslts.append(rslt)

    num_cpu = cpu_count()
    pool = Pool(min(num_cpu, num_workers))

    jobs = []
    videoids = fetch_videoids(phase_key)

    for videoid in videoids:
        # net = rslt_wrapper(phase_key, videoid,
                           # logging.getLogger(str(videoid)))
        # _append_rslt(net)
        jobs.append(
            pool.apply_async(
                rslt_wrapper,
                args=(phase_key, videoid, logging.getLogger(str(videoid))),
                callback=_append_rslt
            )
        )

    pool.close()
    pool.join()

    return pred_rslts


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
        parser.add_argument('--num_workers', type=int, default=16)

        args = parser.parse_args()

        config_path = osp.join(cfg.EXP_DIR, args.dataset,
                               args.expname, 'config.yml')

        # Init
        cfg_from_file(config_path)
        args.phase = args.phase.upper()
        setup(args.phase, args.dataset, args.expname, args.rsltname)

        # Check whether specified <pretrained> when testing:
        # a work-around for conditional required argument.
        if args.phase == 'TEST':
            assert args.pretrained != 'default.caffemodel', \
                'Speicify pretrained model when testing.'
            cfg.SNAPSHOT_PATH = osp.join(cfg.LOCAL_SNAPSHOT_PATH,
                                         args.pretrained)

        caffe.init_glog(osp.join(cfg.LOG_PATH, '{}{}.'.format(args.phase,
                                                              cfg.INFIX)))

        pred_rslts = proc_prll(args.phase, args.num_workers)
        dump_rslt(args.phase, pred_rslts)
