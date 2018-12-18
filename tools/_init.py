import os
import sys

import os.path as osp

from contextlib import contextmanager


############################################################
# Setup path
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


curdir = osp.dirname(__file__)

lib_path = osp.join(curdir, '..', 'lib')
add_path(lib_path)


############################################################
# Import modules from the lib
from config import cfg
from utils.ops import may_create
from utils.proto import prototxt_from_template


@contextmanager
def workenv():
    olddir = os.getcwd()
    os.chdir(osp.join(curdir, '..'))
    try:
        yield
    finally:
        os.chdir(olddir)


def setup(phase_key, dataset, expname, rsltname):
    '''Setup paths & general args after possible merge from config file.'''
    # Save args to config
    cfg.DATASET = dataset
    cfg.EXP = expname

    cfg.NUM_CLASSES = {
        'TH14': 20,
        'AN': 100,
    }[cfg.DATASET]

    # AN.train == TH14.val; AN.val == TH14.test
    # if cfg.DATASET == 'AN':
        # cfg[phase_key].STAGE = {
            # 'val': 'train',
            # 'test': 'val',
            # 'train': 'train',
            # 'val': 'val',
        # }[cfg[phase_key].STAGE]

    # Setup <infix> first, resulting in
    #   '' => ''; 'infix' => '.infix' so that we can uniformly insert it.
    ret_infix = cfg.INFIX if not cfg.INFIX.startswith('.') else cfg.INFIX[1:]
    ret_infix = '' if ret_infix == '' else '.{}'.format(ret_infix)

    cfg.INFIX = ret_infix

    # Setup <viz_folder> name
    norm_str = 'normed' if cfg.FEAT.NORM else 'unnormed'

    avt_str = {
        True: '{avt}',
        False: '{avt}{trh}'
    }[cfg.FEAT.THRESH is None].format(avt=cfg.FEAT.ACTIVATION,
                                      trh=cfg.FEAT.THRESH)

    cfg.VIZ.FOLDER_NAME = '{}_{}_{}_{}'.format(cfg[phase_key].STAGE, cfg.FEAT.MODE,
                                               norm_str, avt_str)

    if not cfg.VIZ.FIX_WIDTH:
        cfg.VIZ.FOLDER_NAME += '_fixwidth'

    # Then several paths: <proto>, <log>, <local_snapshots>, <viz>
    cfg.EXP_PATH = osp.join(cfg.EXP_DIR, cfg.DATASET, cfg.EXP)

    cfg.PROTO_PATH = osp.join(cfg.EXP_PATH, 'proto')
    cfg.LOG_PATH = osp.join(cfg.EXP_PATH, 'log')
    cfg.LOCAL_SNAPSHOT_PATH = osp.join(cfg.EXP_PATH, 'snapshot')
    # Example: exp/TH14/experiment100/val_mul_normed_relu10_fixwidth
    cfg.VIZ_PATH = osp.join(cfg.EXP_PATH, cfg.VIZ.FOLDER_NAME)
    cfg.RSLT_PATH = osp.join(cfg.EXP_PATH, 'rslt')

    path2check = [cfg.PROTO_PATH, cfg.LOG_PATH, cfg.LOCAL_SNAPSHOT_PATH,
                  cfg.VIZ_PATH, cfg.RSLT_PATH]
    map(may_create, path2check)

    cfg.SL_PATH = osp.join(cfg.PROTO_PATH,
                           'solver{}.prototxt'.format(cfg.INFIX))
    cfg.TR_PATH = osp.join(cfg.PROTO_PATH,
                           'train{}.prototxt'.format(cfg.INFIX))
    # Currently we share the prototxt between training and testing.
    cfg.TE_PATH = cfg.TR_PATH

    cfg.SNAPSHOT_PATH = osp.join(cfg.LOCAL_SNAPSHOT_PATH, {
        True: rsltname.replace('.pc', '.caffemodel'),
        False: '{}_iter{}.caffemodel'.format(rsltname, cfg.MAX_ITER)
    }[rsltname.endswith('.pc')])

    # Setup `videoids_lst` template.
    cfg.DSPEC.VID_LST = osp.join(cfg.DATA_DIR, cfg.DATASET, '{stage}_videoid.lst')
    # Specify training input.
    cfg[phase_key].DATA_PATH = osp.join(cfg.DATA_DIR, cfg.DATASET,
                                        cfg[phase_key].DATA_FILE)

    phase_ = phase_key.lower() + '.'
    # Processing rsltname in following logic in order:
    #   (1) rsltname should start with '<phase>.';
    #   (2) rslname with '.pc' should be directly used;
    #   (3) otherwise it should be recorded with the iteration.
    if not rsltname.startswith(phase_):
        rsltname = phase_ + rsltname

    # Finally the result pickle file.
    cfg[phase_key].RSLT_PATH = osp.join(cfg.RSLT_PATH, {
        True: rsltname,
        False: '{}_iter{}.pc'.format(rsltname, cfg.MAX_ITER)
    }[rsltname.endswith('.pc')])

    # Generate prototxt from template
    prototxt_from_template()
